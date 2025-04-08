import numpy
import torch
import json

# from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

import numpy
from typing import List, Dict

from sklearn.preprocessing import OrdinalEncoder

from log_generators.data.dataset import USSEnterpriseSystemsDataset
from log_generators.generators.uss_enterprise import USSEnterpriseDiagnosticGenerator

def compute_metrics(eval_pred):
    """Custom metrics for multi-label classification."""
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    # Calculate precision, recall, F1
    precision = numpy.logical_and(preds, labels).sum() / preds.sum()
    recall = numpy.logical_and(preds, labels).sum() / labels.sum()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def predict(note, model, dataset, compute_device, threshold=0.5):
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
    inputs = dataset.tokenizer(
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
    predicted_labels = dataset.label_encoder.inverse_transform(labels)

    return {
        'systems': predicted_labels,
        'raw_output': probs
    }


if __name__ == '__main__':
   
    base_model_name = 'bert-base-uncased'
     
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    dataset = USSEnterpriseSystemsDataset(
        generator=USSEnterpriseDiagnosticGenerator(refine_with_deepseek=True),
        config = uss_enterprise_systems_info,
    ) 
    train_dataset, val_dataset = dataset, None
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
            num_labels=dataset.n_labels,
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
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=5,
        weight_decay=0.001,
        logging_dir='./logs',
        logging_steps=1,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 5. Predict
    test_note = "Heisenberg compensator degradation with plasma eddies"
    prediction = predict(test_note, model, dataset, compute_device)

    print("Diagnostic Analysis:")
    print(f"Note: {test_note}")
    for system, prob in zip(prediction['systems'], prediction['raw_output']):
        print(f"- {system}: {prob:.2%} probability")
