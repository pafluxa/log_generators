import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
import pandas as pd

from log_analyzer.generators.uss_enterprise import EnterpriseDiagnosticGenerator


class EnterpriseDiagnosticClassifier:
    """
    Fine-tunes BERT with LoRA to classify USS Enterprise system/subsystem failures from diagnostic notes.

    Features:
    - Processes diagnostic notes into system/subsystem probabilities
    - Uses LoRA for efficient fine-tuning
    - Handles multi-label classification
    - Maintains fixed system/subsystem ontology
    """

    def __init__(self, systems_dict: dict):
        """
        Initialize classifier with system configurations.

        Args:
            systems_dict: Complete dictionary of all ship systems and subsystems
        """
        self.systems = systems_dict
        self._initialize_ontology()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(self.all_labels),
            problem_type="multi_label_classification"
        )

        # Initialize LoRA
        self._setup_lora()
        self.model.to(self.device)

    def _initialize_ontology(self):
        """Create fixed list of all system::subsystem pairs."""
        self.all_labels = []
        for system, config in self.systems.items():
            for subsystem in config['subsystems']:
                self.all_labels.append(f"{system}::{subsystem}")

        # Create label binarizer
        self.mlb = MultiLabelBinarizer(classes=self.all_labels)
        self.mlb.fit([self.all_labels])  # Fit on all possible labels

    def _setup_lora(self, r=8, lora_alpha=16, lora_dropout=0.1):
        """Configure LoRA adapter layers."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query", "value"]  # Apply to attention layers
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def preprocess_data(self, notes: list, labels: list):
        """
        Prepare training data.

        Args:
            notes: List of diagnostic notes
            labels: List of lists containing system::subsystem strings
        Returns:
            HuggingFace Dataset ready for training
        """
        # Tokenize text
        encodings = self.tokenizer(
            notes,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        # Convert labels to multi-hot vectors
        binary_labels = self.mlb.transform(labels)

        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': binary_labels.astype(np.float32)
        })

        return dataset

    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=8):
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
            evaluation_strategy="epoch" if val_dataset else "no",
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
        """Custom metrics calculation for multi-label classification."""
        logits, labels = eval_pred
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5

        # Calculate precision, recall, F1
        precision = (preds & labels).sum() / preds.sum()
        recall = (preds & labels).sum() / labels.sum()
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
            threshold: Probability threshold for considering a prediction positive
        Returns:
            dict: {
                'systems': list of predicted system::subsystem pairs,
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

    def save_model(self, path):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, systems_dict, path):
        """Load saved model."""
        instance = cls(systems_dict)
        instance.model = BertForSequenceClassification.from_pretrained(path)
        instance.tokenizer = BertTokenizer.from_pretrained(path)
        instance.model.to(instance.device)
        return instance


# Example Usage
if __name__ == "__main__":
    # 1. Prepare your training data (example format)
    generator = EnterpriseDiagnosticGenerator()
    reports = [rep = generator.generate_full_report()in for _ in range(100)]
    train_data = [{"note": rep['note'], "systems:" rep['systems']} for rep in reports]
    # [
    #     {
    #         "note": "Unstable plasma conduit in sector 5 with secondary gravimetric shear",
    #         "systems": ["warp_core::plasma_conduit", "deflector_dish::graviton_emitter"]
    #     },
    #     # ... more examples
    # ]

    # 2. Initialize classifier with your systems dictionary
    classifier = EnterpriseDiagnosticClassifier(systems)

    # 3. Prepare datasets
    notes = [item["note"] for item in train_data]
    labels = [item["systems"] for item in train_data]
    train_dataset = classifier.preprocess_data(notes, labels)

    # 4. Train the model
    classifier.train(train_dataset, epochs=3)

    # 5. Save the model
    classifier.save_model("./enterprise_diagnostic_bert")

    # 6. Make predictions
    test_note = "Pattern degradation in Heisenberg compensator with secondary plasma eddies"
    prediction = classifier.predict(test_note)

    print("\nDiagnostic Report Analysis:")
    print(f"Note: {test_note}")
    print("\nPredicted System Failures:")
    for system, prob in zip(prediction['systems'], prediction['probabilities']):
        print(f"- {system}: {prob:.2%} probability")
