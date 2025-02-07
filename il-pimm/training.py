import os

import fire
import lightning as L
import torch
from data_utils import PhishingDataModule, PersonalizedPhishingDataModule
from lightning.pytorch import loggers, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from transformer import Classifier, ShallowTransformer, PersonalizedClassifier, PersonalizedShallowTransformer

from utils import clean_dict, make_exp_dir

# personalization -> personal parameters per person (doesn't work well with phishing data)

def main(
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 3,
    seq_length: int = 256,
    dropout: float = 0.1,
    save_top_k: int = 10,
    devices: int = "auto",
    root_dir: str = "/common/home/users/s/shashankc/scratchDirectory",
    data_dir: str = "./data/phishing_encoding",
    seed: int = 42,
    wandb_logger: bool = False,
    run_notes: str = "",
    personalize: bool = False,
    personalization_type: str = "none",
):
    # Set up
    config = clean_dict(locals())
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    exp_dir = make_exp_dir(root_dir)

    # Load the data
    if not personalize:
        dm = PhishingDataModule(batch_size=batch_size, data_dir=data_dir)
    else:
        dm = PersonalizedPhishingDataModule(batch_size=batch_size, data_dir=data_dir)
    dm.setup("fit")

    # Checkpoint & logs
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="epoch_{epoch:02d}-val_acc_{val/acc:.2f}",
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        every_n_epochs=100,
    )
    if wandb_logger:
        logger = loggers.WandbLogger(
            name=os.path.basename(exp_dir),
            project="phishing",
            save_dir=exp_dir,
            config=config,
            notes=run_notes,
        )
    else:
        logger = loggers.TensorBoardLogger(exp_dir)

    # Instantiate the model
    if not personalize:
        if 'iag' in data_dir:
            key_embedding_size = 256
            value_embedding_size = 256
        else:
            key_embedding_size = dm.emb_size
            value_embedding_size = dm.emb_size
        model = Classifier(
            ShallowTransformer(
                num_classes=dm.num_classes,
                emb_size=dm.emb_size,
                seq_length=seq_length,
                dropout=dropout,
                key_hidden_size=dm.emb_size,
                value_hidden_size=dm.emb_size,
            ),
            lr=lr,
        )
    else:
        model = PersonalizedClassifier(
            PersonalizedShallowTransformer(
                num_classes=dm.num_classes,
                emb_size=dm.emb_size,
                seq_length=seq_length,
                dropout=dropout,
                key_hidden_size=dm.emb_size,
                value_hidden_size=dm.emb_size,
                n_persons=dm.n_persons,
                personalization_type=personalization_type,
            ),
            lr=lr,
        )

    # Train the model
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=devices,
    )
    trainer.fit(model, datamodule=dm)

    # Test the model (best chkpt is not stored, use eval.py for evaluation)
    # dm.setup("test")
    # trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    fire.Fire(main)
