from typing import Dict, List, Tuple

import os
import glob
import json
from pathlib import Path

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as papq

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer

from sklearn.preprocessing import MultiLabelBinarizer

from log_generators.generators.uss_enterprise import USSEnterpriseDiagnosticGenerator


class USSEnterpriseSystemsDataset(IterableDataset):

    label_encoder: MultiLabelBinarizer
    n_labels: int
    run_id: str
    base_path: str
    model_name: str

    def __init__(self,
                 base_path: str,
                 model_name: str,
                 run_id: str,
                 config: Dict[str, List[Dict[str, str]]],
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 chunk_size: int = 128,
                 offset: int = 0):

        self.run_id = run_id
        self.base_path = base_path
        self.model_name = model_name
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        if tokenizer_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # create ordinal encoding of labels
        self.labels_as_txt = list(config.keys())
        self._fit_label_encoder()
        self.start = offset
        self.end = self.start + chunk_size

    def _fit_label_encoder(self):

        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit([self.labels_as_txt, ])
        self.n_labels = len(self.label_encoder.classes_)
        print(f"[DEBUG]  found {self.n_labels} unique systems.")

    def _load_chunk(self) -> Tuple[List[str], List[str]]:

        notes: List[str] = []
        systems: List[str] = []
        for id in tqdm(range(self.start, self.end)):
            with open(f"./txt/{self.model_name}/{self.run_id}/notes/{id:06d}.txt", 'r') as f:
                note = f.read().strip()
                notes.append(note)
            # read affected system list
            with open(f"./txt/{self.model_name}/{self.run_id}/systems/{id:06d}.txt", 'r') as f:
                labeled_systems = []
                affections = f.read().strip().split(';')
                for system_subsystem in affections:
                    labeled_systems.append(system_subsystem.split('::')[0])
                systems.append(labeled_systems)

        assert len(notes) == len(systems), f"Different number counts between notes ({len(notes)}) and systems ({len(systems)})."

        return notes, systems

    def __iter__(self):

        notes, systems = self._load_chunk()
        # represent systems using multi-hot encoding
        labels = self.label_encoder.transform(systems)
        labels = torch.tensor(labels, dtype=torch.float32)
        # tokenize notes
        tokenized = self.tokenizer(notes,
            # truncation=True,
            # padding="max_length",
            return_tensors="pt"
        )
        for idx in range(self.chunk_size):
            yield (
                tokenized['input_ids'][idx],
                tokenized['attention_mask'][idx],
                labels[idx]
            )

        self.start = self.start + self.chunk_size
        self.end = self.start + self.chunk_size

if __name__ == '__main__':

    batch_size = 8
    max_length = 512
    n_systems = 0

    with open('log_generators/configs/uss_enterprise.json', 'r') as f:
        config = json.loads(f.read())
    uss_enterprise_systems_info = config

    model_name = "deepseek-r1:32b"
    run_id = 'f4c7c5e7'
    dataset = USSEnterpriseSystemsDataset(
        base_path = './txt',
        model_name = model_name,
        run_id = run_id,
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

    table_path = Path(f"./tokenized/{model_name}/{run_id}/").mkdir(parents=True, exist_ok=True)
    print(table_path)
    papq.write_table(
        patab_data,
        f"tokenized/{model_name}/{run_id}/uss_enterprise_logs.parquet",
        compression=None
    )

    test = papq.read_table(f'tokenized/{model_name}/{run_id}/uss_enterprise_logs.parquet')
    print(test.to_pandas())
