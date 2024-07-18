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
            data_path = os.path.join(data_dir, "phishing_train.pkl")
        else:
            data_path = os.path.join(data_dir, "phishing_test.pkl")
        content_emd_path = os.path.join(data_dir, "content_emd.pkl")
        memory_emd_path = os.path.join(data_dir, "memory_emd.pkl")

        self.data_dir = data_dir
        self.train = train
        raw_data = pd.read_pickle(data_path)
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


class IDGDataset(Dataset):
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
        self.data = raw_data[["row_idx", "memory_idx", "Action"]].rename(
            columns={"Action": "label"}
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


class IDGDataModule(L.LightningDataModule):
    num_classes = 2

    def __init__(
        self, data_dir: str = "/careAIDrive/common/phishing/IDG-emd", batch_size: int = 32
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
            train_val = IDGDataset(self.data_dir, train=True)
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
            self.test = IDGDataset(self.data_dir, train=False)
            self.emb_size = self.test[0][0].shape[-1]

        if stage == "predict":
            self.predict = IDGDataset(self.data_dir, train=False)
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


class TweetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        test_split=0.1,
        val_split=0.1,
        seed=42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.seed = seed
        self.num_workers = 10

    def prepare_data(self):
        dataset = NlpDataset(self._get_twitter_airline_dataset())
        # tokenize the data
        dataset = self._tokenize(dataset)
        l_train = len(dataset) - int(len(dataset) * self.test_split)
        l_test = len(dataset) - l_train
        self.train_dataset, self.test_data = random_split(
            dataset,
            [l_train, l_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def setup(self, stage: str):
        if stage == "fit":
            l_train = len(self.train_dataset) - int(
                len(self.train_dataset) * self.val_split
            )
            l_val = len(self.train_dataset) - l_train
            train_data, val_data = random_split(
                self.train_dataset,
                [l_train, l_val],
                generator=torch.Generator().manual_seed(self.seed),
            )
            self.train = NlpDataset(train_data)
            self.val = NlpDataset(val_data)
        if stage == "test":
            self.test = self.test_data
        if stage == "predict":
            self.predict = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=pad_batch,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val, collate_fn=pad_batch, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, collate_fn=pad_batch, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.predict, collate_fn=pad_batch, num_workers=self.num_workers
        )

    def _get_twitter_airline_dataset(
        self,
        brand: list = [],
        sentiments: str = ["neutral", "positive", "negative"],
        sentiment_confidence_th: float = 0.5,
    ):
        examples = []
        with open(self.data_dir + "/" + "Tweets.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            header = next(csv_reader)

            for line in csv_reader:
                target = line[header.index("airline_sentiment")]
                airline_sentiment_conf = float(
                    line[header.index("airline_sentiment_confidence")]
                )
                airline = line[header.index("airline")]
                source = line[header.index("text")]

                # select the data that fulfills the selected parameters
                if (
                    (target in sentiments)
                    and (
                        airline_sentiment_conf >= sentiment_confidence_th
                        and airline_sentiment_conf <= 1
                    )
                    and (airline in brand if len(brand) != 0 else True)
                    and (source != "")
                ):
                    examples.append({"source": source, "target": target})

        return examples

    def _tokenize(self, data):
        self.text_encoder = MosesEncoder(data["source"])
        self.label_encoder = LabelEncoder(data["target"])

        # tokenize sources and targets
        labels = []
        sources = []
        for source, label in zip(data["source"], data["target"]):
            labels.append(self.label_encoder.encode(label))
            sources.append(self.text_encoder.encode(source))
        data["target"] = labels
        data["source"] = sources

        self.vocab_size = self.text_encoder.vocab_size
        self.num_classes = self.label_encoder.vocab_size - 1
        return data


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


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

    dm = IDGDataModule(batch_size=32)
    dm.prepare_data()
    dm.setup("fit")
    dl = dm.train_dataloader()
    tok, mask, target = next(iter(dl))
    print("Input shape:", tok.shape)
    print("Num Classes:", dm.num_classes)
    print("Targets:", target)

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
