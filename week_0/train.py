import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel



def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics")
    trainer = pl.Trainer(
        accelerator='gpu',
        # gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=1,
        fast_dev_run=False,
        logger=wandb_logger,
        # pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
