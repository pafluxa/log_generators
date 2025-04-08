import numpy
import torch
import json

import pyarrow as pa
import pyarrow.parquet as papq

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from peft import LoraConfig, get_peft_model, TaskType

import numpy
from typing import List, Dict

from sklearn.preprocessing import OrdinalEncoder

from log_generators.data.dataset import USSEnterpriseSystemsDataset
from log_generators.generators.uss_enterprise import USSEnterpriseDiagnosticGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(p: EvalPrediction):
    """Custom metrics for multi-label classification."""
    logits = p.predictions
    # logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = p.label_ids
    # Calculate precision, recall, F1
    precision = numpy.logical_and(preds, labels).sum() / preds.sum()
    recall = numpy.logical_and(preds, labels).sum() / labels.sum()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        'precision': precision.item(),
       'recall': recall.item(),
       'f1': f1.item()
   }

def predict(note, model, label_encoder, tokenizer, compute_device, threshold=0.5):
    """
    Predict system/subsystem probabilities for a diagnostic note.

    Args:
        note: Diagnostic note text
        threshold: Probability threshold for positive prediction
    Returns:
        dict: {
            'systems': predicted system::subsystem pairs,
            'probabilities': corresponding probabilities,
            'raw_output': full probability vector
        }
    """
    # Tokenize input
    inputs = tokenizer(
        note,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(compute_device)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Get predictions above threshold
    above_threshold = probs > threshold
    labels = numpy.zeros_like(probs, dtype=numpy.int64)
    labels[above_threshold] = 1
    labels = numpy.expand_dims(labels, axis=0)
    predicted_labels = label_encoder.inverse_transform(labels)

    probs = probs[above_threshold]
    sorting_idx = numpy.argsort(probs)
    
    return {
        'systems': predicted_labels[0],
        'probabilities': probs,
        'sorting_index': sorting_idx
    }

def dataset_to_hf_dataset(names, chunk_sizes):
    
    max_length = 512
    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    for name, chunk_size in zip(names, chunk_sizes):
        dataset = USSEnterpriseSystemsDataset(
            generator=USSEnterpriseDiagnosticGenerator(refine_with_deepseek=False),
            config = uss_enterprise_systems_info,
            chunk_size=chunk_size
        )
        n_systems = dataset.n_labels

        t1 = []
        t2 = []
        t3 = []
        for input_ids, attn_mask, enc_lbl in dataset:
            t1.append([input_ids.numpy(),])
            t2.append([attn_mask.numpy(),])
            t3.append([enc_lbl.numpy(),])
        c1 = pa.chunked_array(t1)
        c2 = pa.chunked_array(t2)
        c3 = pa.chunked_array(t3)
        patab_data = pa.table(
            [c1, c2, c3], 
            names=['input_ids', 'attention_mask', 'labels'],
        )
        papq.write_table(
            patab_data, 
            f"./tokenized/uss_enterprise_logs/{name}.parquet", 
            compression=None
        )
    
    return n_systems, dataset.labels_as_txt, dataset.label_encoder, dataset.tokenizer

if __name__ == '__main__':
   
    base_model_name = 'bert-base-uncased'
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_systems, label_names, label_encoder, tokenizer = dataset_to_hf_dataset(["train", "test"], [128, 32])
    
    dataset = load_dataset("parquet", 
                data_dir="./tokenized/uss_enterprise_logs/", 
                data_files={
                    'train': 'train.parquet',
                    'test': 'test.parquet',
                })
    train_dataset, eval_dataset = dataset['train'], dataset['test']
     
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=n_systems,
        problem_type="multi_label_classification"
    )
    model.to(compute_device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules='all-linear',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    """Fine-tune the model with LoRA."""
    training_args = TrainingArguments(
        label_names=["labels"],
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=4,
        weight_decay=0.001,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    # 5. Evaluate the fine-tuned model
    results = trainer.evaluate()

    # Print evaluation results
    print(results)

    # 6. Predict
    test_note = "Dilithium matrix vectors out of range with indications of backwards antimatter flow."
    prediction = predict(test_note, model, label_encoder, tokenizer, compute_device)
    systems, probs = prediction['systems'], prediction['probabilities']

    print("Diagnostic Analysis:")
    print(f"Note: {test_note}")
    for idx in prediction['sorting_index']:
        print(f"- {systems[idx]}: {probs[idx]:.2%} probability")
