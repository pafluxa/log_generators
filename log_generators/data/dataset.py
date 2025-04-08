from typing import Dict, List, Tuple

import numpy

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from sklearn.preprocessing import MultiLabelBinarizer

from log_generators.generators.uss_enterprise import USSEnterpriseDiagnosticGenerator


class USSEnterpriseSystemsDataset(IterableDataset):
    
    label_encoder: MultiLabelBinarizer
    n_labels: int
    
    def __init__(self, 
                 generator: USSEnterpriseDiagnosticGenerator,
                 config: Dict,
                 tokenizer_name: str = 'bert-base-uncased', 
                 max_length: int = 512):
        """
        Args:
            texts (list): List of symptom description strings
            system_pairs (list): List of system/subsystem strings in "system::subsystem" format
            tokenizer_name (str): Name/path of Hugging Face tokenizer
            max_length (int): Maximum sequence length for tokenization
        """
        self.gen = generator
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.chunk_size = 8
        # create ordinal encoding of labels
        self.labels_as_txt = list(config.keys())
        self._fit_label_encoder()
        
    def _fit_label_encoder(self):
        
        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.labels_as_txt, ])
        self.n_labels = len(self.label_encoder.classes_)
        print(f"[DEBUG]  found {self.n_labels} unique systems.")

    def _generate_and_parse_reports(self, chunksize: int) -> Dict[str, List[str]]:
        
        notes = []
        systems = [] 
        for _ in range(chunksize):
            data = self.gen.generate_report()
            note_txt = data['note']
            affections_raw = data['systems']
            affections = affections_raw.split(';')
            affected_systems = [asys.split('::')[0] for asys in affections]
        
            notes.append(note_txt) 
            systems.append(affected_systems)
            
        return {'notes': notes, 'systems': systems}

    def __iter__(self):
        
        # generate chunk of reports
        data = self._generate_and_parse_reports(self.chunk_size)
        notes = data['notes']
        affected_systems = data['systems']
        # encode affected systems as ordinals
        multihot_labels = self.label_encoder.transform(affected_systems)
        # tokenize notes
        tokenized = self.tokenizer(notes,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        for idx, labels in enumerate(multihot_labels):
            # multi-hot vector encodes labels
            encoded_labels = torch.tensor(labels, dtype=torch.float32)
            yield {
                'input_ids': tokenized['input_ids'][idx],
                'attention_mask': tokenized['attention_mask'][idx],
                'labels': encoded_labels,
                'text': notes[idx]
            }
            
if __name__ == '__main__':
    
    import json
    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    dataset = USSEnterpriseSystemsDataset(
        generator=USSEnterpriseDiagnosticGenerator(refine_with_deepseek=True),
        config = uss_enterprise_systems_info,
    )

    counter = 0
    for entry in dataset:
        print(entry['text'])
        counter += 1
        if counter > 8:
            break