import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from config.config import load_config
from src.data.datamodule import DyanmicsDataset
from src.data.datasplit import DataSplitUtils
from src.trainer.simple_trainer import TrainerModule

wandb_logger = WandbLogger(project="229-project")


def run_experiment():

    config = load_config("config/debug.yaml")
    pl.seed_everything(config["seed"])
    # Read dataset
    data_class = DataSplitUtils(config["data"]["dataset_path"])
    _ = DataSplitUtils.load_data(data_class, config["data"]["modalities"].keys())
    # split into trian val test
    data_train, data_test, data_val = data_class.random_train_test_split(
        test_size=config["train"]["train_split"]
    )
    train_dataset = DyanmicsDataset(config, data_train, True)
    test_dataset = DyanmicsDataset(config, data_test, False)
    val_dataset = DyanmicsDataset(config, data_val, False)
    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    print(f"{len(val_dataset)=}")
    train_loader = train_dataset.get_dataloader()
    test_loader = test_dataset.get_dataloader()
    val_loader = val_dataset.get_dataloader()
    model = TrainerModule(config, 16, 2)  # remove hardcoding, wrap this in a class
    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epochs"],
        logger=wandb_logger,
        accelerator="gpu",
        fast_dev_run=config["fast_dev_run"],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    run_experiment()
    wandb.finish()
