import numpy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from datasets import Dataset
from typing import List, Dict

from log_analyzer.generators.uss_enterprise import EnterpriseDiagnosticGenerator


class EnterpriseDiagnosticClassifier:
    """
    Fine-tunes BERT with LoRA to classify USS Enterprise system/subsystem failures.
    Uses manual multi-hot encoding for multi-label classification.
    """

    def __init__(self, systems_dict: Dict):
        """
        Initialize classifier with ship systems configuration.

        Args:
            systems_dict: Complete dictionary of all ship systems and subsystems
                         Format: {"system_name": {"subsystems": [...], ...}, ...}
        """
        self.systems = systems_dict
        self._initialize_ontology()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT components
        self.tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2b") #BertTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = AutoModelForSequenceClassification.from_pretrained("google/gemma-2b"
             num_labels=len(self.all_labels),
             problem_type="multi_label_classification"
        )

        # Initialize LoRA
        self._setup_lora()
        self.model.to(self.device)

    def _initialize_ontology(self):
        """Create fixed list of all system::subsystem pairs with indices."""
        self.all_labels = []

        # Build list of all possible system::subsystem pairs
        for system_name, system_config in self.systems.items():
            for subsystem in system_config['subsystems']:
                self.all_labels.append(f"{system_name}::{subsystem}")

        # Create mapping dictionaries
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def _setup_lora(self, r=8, lora_alpha=16, lora_dropout=0.1):
        """Configure LoRA adapter layers."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=[
                "all-linear",
            ]
            #     "mlp.dense_4h_to_h",
            # "k_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj""query", "value"]
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _labels_to_tensor(self, label_lists: List[List[str]]) -> torch.Tensor:
        """Convert list of label lists to multi-hot tensor."""
        batch_size = len(label_lists)
        tensor = torch.zeros((batch_size, len(self.all_labels)), dtype=torch.float32)

        for i, labels in enumerate(label_lists):
            for label in labels:
                if label in self.label_to_idx:
                    tensor[i, self.label_to_idx[label]] = 1.0
        return tensor

    def preprocess_data(self, notes: List[str], labels: List[List[str]]):
        """
        Prepare training data.

        Args:
            notes: List of diagnostic note texts
            labels: List of lists containing system::subsystem strings
        Returns:
            HuggingFace Dataset ready for training
        """
        # Tokenize text
        encodings = self.tokenizer(
            notes,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )

        # Convert labels to multi-hot tensors
        label_tensors = self._labels_to_tensor(labels)

        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': label_tensors
        })

    def train(self, train_dataset, val_dataset=None, epochs=50, batch_size=8):
        """Fine-tune the model with LoRA."""
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )

        trainer.train()

    def _compute_metrics(self, eval_pred):
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

    def predict(self, note: str, threshold=0.5):
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
        inputs = self.tokenizer(
            note,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Get predictions above threshold
        above_threshold = probs > threshold
        predicted_indices = np.where(above_threshold)[0]
        predicted_labels = [self.all_labels[i] for i in predicted_indices]
        predicted_probs = [probs[i] for i in predicted_indices]

        return {
            'systems': predicted_labels,
            'probabilities': predicted_probs,
            'raw_output': probs
        }

    def save_model(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, systems_dict: Dict, path: str):
        """Load saved model."""
        instance = cls(systems_dict)
        instance.model = BertForSequenceClassification.from_pretrained(path)
        instance.tokenizer = BertTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        return instance


# Example Usage
if __name__ == "__main__":

    generator = EnterpriseDiagnosticGenerator()
    systems = generator._initialize_system_templates()

    # 1. Initialize with your systems dictionary
    classifier = EnterpriseDiagnosticClassifier(systems)

    # 2. Prepare training data (example)
    reports = [generator.generate_report() for _ in range(50)]
    train_data = [{"note": rep['note'], "systems": rep['systems']} for rep in reports]

    # 3. Preprocess data
    notes = [item["note"] for item in train_data]
    labels = [item["systems"] for item in train_data]
    train_dataset = classifier.preprocess_data(notes, labels)
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=0.3, shuffle=True).values()
    # 4. Train
    classifier.train(train_dataset, val_dataset=val_dataset, epochs=100)

    # 5. Predict
    test_note = "Heisenberg compensator degradation with plasma eddies"
    prediction = classifier.predict(test_note, threshold=0.2)

    print("Diagnostic Analysis:")
    print(f"Note: {test_note}")
    for system, prob in zip(prediction['systems'], prediction['probabilities']):
        print(f"- {system}: {prob:.2%} probability")
