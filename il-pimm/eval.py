import os

import fire
import lightning as L
import torch
from data_utils import PhishingDataModule, PersonalizedPhishingDataModule
from lightning.pytorch import loggers, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from transformer import Classifier, ShallowTransformer, PersonalizedClassifier, PersonalizedShallowTransformer
import re

def main(
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_length: int = 256,
    dropout: float = 0.1,
    data_dir: str = "./data/phishing_encoding",
    seed: int = 42,
    personalize: bool = False,
    personalization_type: str = "none",
    checkpoint_dir='/common/home/users/s/shashankc/code/Phishing/Hybrid/phishing-ai/shallow_transformer/results/phishing/5010/Personalized/009_2025-01-17_16-53-39/checkpoints',
):
    
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Load the data
    if not personalize:
        dm = PhishingDataModule(batch_size=batch_size, data_dir=data_dir)
    else:
        dm = PersonalizedPhishingDataModule(batch_size=batch_size, data_dir=data_dir)
    dm.setup("fit")

    pattern = r"epoch_(\d+)-val_acc_([0-9]+\.[0-9]+)"

    test_accs = {}
    val_accs = {}
    # Iterate over files in the checkpoint directory
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            # Use the checkpoint_path as needed (e.g., load the checkpoint, perform inference, etc.)
            print("Checkpoint:", checkpoint_path)

            if('last' in checkpoint_path):
                continue
            
            match = re.search(pattern, filename)
            if match:
                epoch_num = int(match.group(1))  # Extract the epoch number
                val_acc = float(match.group(2))  # Extract the validation accuracy value
                print(f"Epoch: {epoch_num}, Validation Accuracy: {val_acc}, Checkpoint: {checkpoint_path}")

            else:
                print('No match')
                break
            
            chkpt_path = checkpoint_path

            if not personalize:
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
            
            trainer = L.Trainer(
                        max_epochs=1,
                        devices="auto",
                    )

            dm.setup("test")
            test_acc = trainer.test(model=model, ckpt_path=chkpt_path, datamodule=dm)[0]['test/acc']
            test_accs[epoch_num] = test_acc
            val_accs[epoch_num] = val_acc

    import matplotlib.pyplot as plt

    d = {}
    for epoch in test_accs.keys():
        d[epoch] = [val_accs[epoch], test_accs[epoch]]
        
    d = dict(sorted(d.items(), key=lambda item: item[0]))
    epochs = list(d.keys())
    val_accs_, test_accs_ = zip(*d.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accs_, label='Validation Accuracy', marker='o')
    plt.plot(epochs, test_accs_, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracies')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)  # Set x-axis ticks to match epoch numbers
    plt.savefig('test/test_results.jpg')
    plt.show()

if __name__ == "__main__":
    main(
        data_dir='',
        checkpoint_dir='',
        personalize=False,
    )