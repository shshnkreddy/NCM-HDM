import os

import fire
import lightning as L
import torch
from data_utils import PhishingDataModule
from lightning.pytorch import loggers, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from transformer import Classifier, CTransformer

from utils import clean_dict, make_exp_dir


def main(
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 3,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_length: int = 256,
    max_pool: bool = False,
    dropout: float = 0.1,
    save_top_k: int = 3,
    devices: int = "auto",
    root_dir: str = "./results/phishing_encoding",
    data_dir: str = "./data/phishing_encoding",
    seed: int = 42,
    wandb_logger: bool = True,
    run_notes: str = "",
):
    # Set up
    config = clean_dict(locals())
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    exp_dir = make_exp_dir(root_dir)

    # Load the data
    dm = PhishingDataModule(batch_size=batch_size, data_dir=data_dir)
    dm.setup("fit")

    # Checkpoint & logs
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="epoch_{epoch:02d}-val_loss_{val/loss:.2f}",
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
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
    model = Classifier(
        CTransformer(
            num_classes=dm.num_classes,
            emb_size=dm.emb_size,
            heads=n_heads,
            depth=n_layers,
            seq_length=seq_length,
            max_pool=max_pool,
            dropout=dropout,
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

    # Test the model
    dm.setup("test")
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    fire.Fire(main)
