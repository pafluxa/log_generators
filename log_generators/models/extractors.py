import numpy
import torch
import json
from pathlib import Path

from torch import nn

import pyarrow as pa
import pyarrow.parquet as papq

from datasets import Dataset, load_dataset

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import numpy
from typing import List, Dict

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from log_generators.data.dataset import USSEnterpriseSystemsDataset


def compute_metrics(p: EvalPrediction):
    """Custom metrics for multi-label classification."""
    logits = p.predictions
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = p.label_ids
    # Calculate precision, recall, F1
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
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

def dataset_to_hf_dataset(tokenizer_name, names, chunk_sizes):

    max_length = 2048
    n_systems = 0

    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    val_dataset = None
    model_name = "deepseek-r1:32b"
    run_id = 'f4c7c5e7'
    offset = 0
    for n, name in zip(chunk_sizes, names):
        dataset = USSEnterpriseSystemsDataset(
            base_path = './txt',
            model_name = model_name,
            tokenizer_name = tokenizer_name,
            run_id = run_id,
            config = uss_enterprise_systems_info,
            chunk_size = n,
            offset = offset,
            max_length = max_length,
        )
        offset = offset + n
        if name == 'validation':
            val_dataset = dataset

        n_systems = dataset.n_labels

        pasch_data = pa.schema([
            ("input_ids", pa.list_(pa.int64(), max_length)),
            ("attention_mask", pa.list_(pa.int64(), max_length)),
            ("labels", pa.list_(pa.float32(), n_systems))
        ])

        t1, t2, t3 = [], [], []
        for input_ids, attn_mask, enc_lbl in dataset:
            t1.append([input_ids.numpy(),])
            t2.append([attn_mask.numpy(),])
            t3.append([enc_lbl.numpy(),])
        c1 = pa.chunked_array(t1)
        c2 = pa.chunked_array(t2)
        c3 = pa.chunked_array(t3)

        patab_data = pa.table(
            [c1, c2, c3],
            names=['input_ids', 'attention_mask', 'labels']
        )

        Path(f"./tokenized/{model_name}/{run_id}/").mkdir(parents=True, exist_ok=True)

        papq.write_table(
            patab_data,
            f"tokenized/{model_name}/{run_id}/{name}.parquet",
            compression=None
        )

    return n_systems, dataset.labels_as_txt, dataset.label_encoder, dataset.tokenizer, val_dataset

class BCETrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # loss_fct = torch.nn.BCEWithLogitsLoss()
        # print(logits[0], labels[0])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':

    base_model_name = 'microsoft/phi-1_5'
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_systems, label_names, label_encoder, tokenizer, val_pt_dataset = \
            dataset_to_hf_dataset(
                base_model_name, 
                ["train", "test", "validation"], 
                [300, 50, 50]
            )

    dataset = load_dataset("parquet",
                data_dir="./tokenized/deepseek-r1:32b/f4c7c5e7",
                data_files={
                    "train": "train.parquet",
                    "test": "test.parquet",
                    "validation": "validation.parquet"})

    train_dataset = dataset["test"]
    test_dataset = dataset["test"]
    validation_dataset = dataset["validation"]

    quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=n_systems,
        problem_type="multi_label_classification",
        torch_dtype=torch.float16,
        # quantization_config=quantization_config
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "dense",
            "o_proj",
            "qkv_proj"
        ],
        init_lora_weights='gaussian',

    )
    model = get_peft_model(model, lora_config)
    model.to(compute_device)

    model.print_trainable_parameters()
    
    # 6. Predict
    print("NO TRAINING")
    print("-" * 80)
    notes, systems = val_pt_dataset._load_chunk()
    for test_note, test_system in zip(notes, systems):
        
        prediction = predict(test_note, model, label_encoder, tokenizer, compute_device)
        predicted_systems, predicted_probs, sorting_index = \
                prediction['systems'], prediction['probabilities'], prediction['sorting_index']
        s_gt = set(test_system)
        s_pr = set(predicted_systems)
        # matches correspond to the intersection of ground-truth with prediction
        s_match = s_gt.intersection(s_pr)
        # misses correspond to the difference between ground-truth and prediction
        s_misses = s_gt.difference(s_pr)

        print("Diagnostic Analysis:")
        print(f"Note: {test_note}")
        print(f"[MATCH] {list(s_match)}")
        print(f"[MISSES] {list(s_misses)}")
        print("Predictions:")
        for idx in sorting_index:
            pred_system = predicted_systems[idx]
            pred_prob = predicted_probs[idx]
            if pred_system in list(s_match):
                print(f"[POSITIVE MATCH] {pred_system} {pred_prob:.2%} probability")
            else:
                print(f"[NEGATIVE MATCH] {pred_system} {pred_prob:.2%} probability")

        print("\n\n")
    print("-" * 80)


    """Fine-tune the model with LoRA."""
    training_args = TrainingArguments(
        label_names=["labels"],
        output_dir='./results',
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.001,
        eval_strategy="epoch",
    )

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 5. Evaluate the fine-tuned model
    results = trainer.evaluate(validation_dataset)

    # Print evaluation results
    print("\n\nEVALUATION RESULTS")
    print(results)
    print("-"* 80)
    print("\n\n")

    # 6. Predict
    notes, systems = val_pt_dataset._load_chunk()
    for test_note, test_system in zip(notes, systems):
        
        prediction = predict(test_note, model, label_encoder, tokenizer, compute_device)
        predicted_systems, predicted_probs, sorting_index = \
                prediction['systems'], prediction['probabilities'], prediction['sorting_index']
        s_gt = set(test_system)
        s_pr = set(predicted_systems)
        # matches correspond to the intersection of ground-truth with prediction
        s_match = s_gt.intersection(s_pr)
        # misses correspond to the difference between ground-truth and prediction
        s_misses = s_gt.difference(s_pr)

        print("Diagnostic Analysis:")
        print(f"Note: {test_note}")
        print(f"[MATCH] {list(s_match)}")
        print(f"[MISSES] {list(s_misses)}")
        print("Predictions:")
        for idx in sorting_index:
            pred_system = predicted_systems[idx]
            pred_prob = predicted_probs[idx]
            if pred_system in list(s_match):
                print(f"[POSITIVE MATCH] {pred_system} {pred_prob:.2%} probability")
            else:
                print(f"[NEGATIVE MATCH] {pred_system} {pred_prob:.2%} probability")

        print("\n\n")
