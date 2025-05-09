import csv
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchnlp.datasets.dataset import Dataset as NlpDataset
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import MosesEncoder
from torchnlp.utils import lengths_to_mask
from torchvision import transforms
from torchvision.datasets import MNIST

class PhishingDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True):
        if train:
            data_path = os.path.join(data_dir, "train.json")
        else:
            data_path = os.path.join(data_dir, "test.json")
        content_emd_path = os.path.join(data_dir, "content_emd.pkl")
        memory_emd_path = os.path.join(data_dir, "memory_emd.pkl")

        self.data_dir = data_dir
        self.train = train
        raw_data = pd.read_json(data_path)
        self.data = raw_data[["row_idx", "memory_idx", "user_action1"]].rename(
            columns={"user_action1": "label"}
        )
        self.content_emd = pd.read_pickle(content_emd_path)
        self.memory_emd = pd.read_pickle(memory_emd_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_idx, memory_indices, label = self.data.iloc[idx]
        input_data = [self.memory_emd[idx] for idx in memory_indices] + [
            self.content_emd[row_idx]
        ]
        return np.array(input_data), label

class PersonalizedPhishingDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True):
        if train:
            data_path = os.path.join(data_dir, "train.json")
        else:
            data_path = os.path.join(data_dir, "test.json")
        content_emd_path = os.path.join(data_dir, "content_emd.pkl")
        memory_emd_path = os.path.join(data_dir, "memory_emd.pkl")

        self.data_dir = data_dir
        self.train = train
        raw_data = pd.read_json(data_path)
        self.data = raw_data[["row_idx", "memory_idx", "user_action1", "Mturk_id"]].rename(
            columns={"user_action1": "label"}
        )
        self.content_emd = pd.read_pickle(content_emd_path)
        self.memory_emd = pd.read_pickle(memory_emd_path)

        full_data = pd.read_json(os.path.join(data_dir, "full_data.jsonl"), lines=True)
        user_index_map = {}
        i = 0
        for index, row in full_data.iterrows():
            if row['Mturk_id'] not in user_index_map:
                user_index_map[row['Mturk_id']] = i
                i += 1

        self.data['user_id'] = self.data['Mturk_id'].apply(lambda x: user_index_map[x])
        self.n_persons = len(user_index_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_idx, memory_indices, label, _, user_id = self.data.iloc[idx]
        input_data = [self.memory_emd[idx] for idx in memory_indices] + [
            self.content_emd[row_idx]
        ]
        return np.array(input_data), label, user_id

class PhishingDataModule(L.LightningDataModule):
    num_classes = 2

    def __init__(
        self, data_dir: str = "./data/phishing_encoding", batch_size: int = 32
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_val = PhishingDataset(self.data_dir, train=True)
            self.train, self.val = random_split(
                train_val,
                [
                    int(len(train_val) * 0.85),
                    len(train_val) - int(len(train_val) * 0.85),
                ],
                generator=torch.Generator().manual_seed(42),
            )
            self.emb_size = train_val[0][0].shape[-1]
        if stage == "test":
            self.test = PhishingDataset(self.data_dir, train=False)
            self.emb_size = self.test[0][0].shape[-1]

        if stage == "predict":
            self.predict = PhishingDataset(self.data_dir, train=False)
            self.emb_size = self.predict[0][0].shape[-1]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=10,
        )

    # NOTE: batch_size is set to 1 for validation and test dataloaders
    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=collate_fn,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=collate_fn,
            num_workers=10,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            collate_fn=collate_fn,
            num_workers=10,
        )


class PersonalizedPhishingDataModule(L.LightningDataModule):
    num_classes = 2

    def __init__(
        self, data_dir: str = "./data/phishing_encoding", batch_size: int = 32
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_val = PersonalizedPhishingDataset(self.data_dir, train=True)
            self.train, self.val = random_split(
                train_val,
                [
                    int(len(train_val) * 0.85),
                    len(train_val) - int(len(train_val) * 0.85),
                ],
                generator=torch.Generator().manual_seed(42),
            )
            self.emb_size = train_val[0][0].shape[-1]
            self.n_persons =  train_val.n_persons
            
        if stage == "test":
            self.test = PersonalizedPhishingDataset(self.data_dir, train=False)
            self.emb_size = self.test[0][0].shape[-1]
            self.n_persons =  self.test.n_persons

        if stage == "predict":
            self.predict = PersonalizedPhishingDataset(self.data_dir, train=False)
            self.emb_size = self.predict[0][0].shape[-1]
            self.n_persons =  self.predict.n_persons

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=personalized_collate_fn,
            num_workers=10,
        )

    # NOTE: batch_size is set to 1 for validation and test dataloaders
    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=personalized_collate_fn,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=personalized_collate_fn,
            num_workers=10,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            collate_fn=personalized_collate_fn,
            num_workers=10,
        )

def collate_fn(batch):
    """
    Custom collate_fn for the phishing dataset.
    """

    def pad_seq(seq, max_len):
        pad_size = max_len - len(seq)
        padding_array = np.zeros((pad_size, seq.shape[1]))
        return np.concatenate((seq, padding_array), axis=0)

    sources, labels = zip(*batch)
    sources_lengths = [len(s) for s in sources]
    max_len = max(sources_lengths)
    source_padded = np.stack([pad_seq(s, max_len) for s in sources], axis=0)
    sources_mask = lengths_to_mask(sources_lengths)
    labels = np.array(labels)

    return (
        torch.tensor(source_padded, dtype=torch.float32),
        sources_mask.unsqueeze(1),
        torch.tensor(labels, dtype=torch.long),
    )

def personalized_collate_fn(batch):
    """
    Custom collate_fn for the phishing dataset.
    """

    def pad_seq(seq, max_len):
        pad_size = max_len - len(seq)
        padding_array = np.zeros((pad_size, seq.shape[1]))
        return np.concatenate((seq, padding_array), axis=0)

    sources, labels, user_ids = zip(*batch)
    sources_lengths = [len(s) for s in sources]
    max_len = max(sources_lengths)
    source_padded = np.stack([pad_seq(s, max_len) for s in sources], axis=0)
    sources_mask = lengths_to_mask(sources_lengths)
    labels = np.array(labels)

    return (
        torch.tensor(source_padded, dtype=torch.float32),
        sources_mask.unsqueeze(1),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(user_ids, dtype=torch.int)
    )



def pad_batch(batch, pad_token=0):
    # Get maximum length of sequences in the batch
    max_len = max(len(data["source"]) for data in batch)

    # Initialize lists to store padded sequences and masks
    input_padded = []
    input_mask = []
    target = []

    # Pad sequences and create masks
    for data in batch:
        # Convert source sequence to tensor
        source_tensor = data["source"]

        # Pad source sequence
        padded_source = torch.cat(
            (
                source_tensor,
                torch.tensor(
                    [pad_token] * (max_len - len(source_tensor)), dtype=torch.int
                ),
            ),
            dim=0,
        )
        input_padded.append(padded_source)

        # Create input mask
        mask = torch.cat(
            (
                torch.ones(len(source_tensor), dtype=torch.bool),
                torch.zeros(max_len - len(source_tensor), dtype=torch.bool),
            ),
            dim=0,
        )
        input_mask.append(mask)

        # Add target
        target.append(data["target"] - 1)

    # Stack tensors along the first dimension
    input_padded = torch.stack(input_padded)
    input_mask = torch.stack(input_mask)
    target = torch.tensor(target, dtype=torch.long)

    return input_padded, input_mask.unsqueeze(1), target


if __name__ == "__main__":
    dm = PersonalizedPhishingDataModule(batch_size=2, data_dir='/careAIDrive/common/phishing/data-related/Phishing-datasets/Emd/5010/')
    dm.setup("fit")
    dl = dm.train_dataloader()
    print(next(iter(dl)))

    # dm = PhishingDataModule(batch_size=2)
    # dm.setup("fit")
    # dl = dm.train_dataloader()
    # print(next(iter(dl)))

    # # test the built-in data module
    # dm = MNISTDataModule(data_dir="./data")
    # dm.prepare_data()
    # dm.setup("fit")
    # dl = dm.train_dataloader()
    # print(next(iter(dl)))

    # dm = TweetDataModule(batch_size=32)
    # dm.prepare_data()
    # dm.setup("fit")
    # dl = dm.train_dataloader()
    # tok, mask, target = next(iter(dl))
    # print("Num Classes:", dm.num_classes)
    # print("Targets:", target)

    # from transformer import CTextTransformer
    # model = CTextTransformer(
    #     vocab_size=dm.vocab_size,
    #     num_classes=dm.num_classes,
    #     emb_size=128,
    #     heads=4,
    #     depth=4,
    #     seq_length=256,
    #     max_pool=False,
    # )
    # print(model.forward(tok, mask))
