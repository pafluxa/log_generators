from typing import Dict, List, Tuple
import json
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as papq

import torch
from torch.utils.data import IterableDataset, DataLoader
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
                 max_length: int = 512,
                 chunk_size: int = 128):
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
        self.chunk_size = chunk_size
        # create ordinal encoding of labels
        self.labels_as_txt = list(config.keys())
        self._fit_label_encoder()
    
    def _fit_label_encoder(self):
        
        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.labels_as_txt, ])
        self.n_labels = len(self.label_encoder.classes_)
        print(f"[DEBUG]  found {self.n_labels} unique systems.")

    def _generate_and_parse_reports(self) -> Dict[str, List[str]]:
        
        notes = []
        systems = [] 
        for _ in tqdm(range(self.chunk_size)):
            data = self.gen.generate_report()
            note_txt = data['note'][0]
            affections_raw = data['systems']
            affections = affections_raw.split(';')
            affected_systems = [asys.split('::')[0] for asys in affections]
            notes.append(note_txt) 
            systems.append(affected_systems)
            
        return notes, systems

    def __iter__(self):
        
        notes, systems = self._generate_and_parse_reports()
        # represent systems using multi-hot encoding
        labels = self.label_encoder.transform(systems)
        labels = torch.tensor(labels, dtype=torch.float32)
        # tokenize notes
        tokenized = self.tokenizer(notes,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        for idx in range(self.chunk_size):
            yield (
                tokenized['input_ids'][idx],
                tokenized['attention_mask'][idx],
                labels[idx]
            )
            
            
if __name__ == '__main__':
    
    batch_size = 8
    max_length = 512
    n_systems = 0
     
    import json
    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    dataset = USSEnterpriseSystemsDataset(
        generator=USSEnterpriseDiagnosticGenerator(refine_with_deepseek=False),
        config = uss_enterprise_systems_info,
    )
    n_systems = dataset.n_labels

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    pasch_data = pa.schema([
        ("input_ids", pa.list_(pa.int64(), max_length)), 
        ("attention_mask", pa.list_(pa.int64(), max_length)),
        ("labels", pa.list_(pa.float32(), n_systems))    
    ])

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
        names=['input_ids', 'attention_mask', 'labels']
    )
    
    papq.write_table(
        patab_data, 
        "tokenized/uss_enterprise_logs.parquet", 
        compression=None
    )
    
    test = papq.read_table('tokenized/uss_enterprise_logs/train.parquet')
    print(test.to_pandas())