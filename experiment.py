import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from config.config import load_config
from src.data.datamodule import DyanmicsDataset
from src.data.datasplit import DataSplitUtils
from src.trainer.simple_trainer import TrainerModule

wandb_logger = WandbLogger(project="229-project")


def visualize(model, loader, title: str):

    # sample random batch from loader
    batch = next(iter(loader))
    state = batch["state"].float()
    actions = batch["action_horizon"].float()
    targets = batch["ground_truth"].float()
    predictions = model(state, actions)
    # plot random sample from batch
    idx = np.random.randint(0, state.shape[0])
    target_xyz = targets[idx, :, :3].detach().cpu().numpy()
    pred_xyz = predictions[idx, :, :3].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        target_xyz[:, 0],
        target_xyz[:, 1],
        target_xyz[:, 2],
        label="Ground Truth",
        marker="o",
        linestyle="-",
    )
    ax.plot(
        pred_xyz[:, 0],
        pred_xyz[:, 1],
        pred_xyz[:, 2],
        label="Predictions",
        marker="x",
        linestyle="--",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.show()


def run_experiment():

    config = load_config("config/debug.yaml")
    pl.seed_everything(config["seed"])
    # Read dataset
    data_class = DataSplitUtils(config["data"]["dataset_path"])
    _ = DataSplitUtils.load_data(data_class, config["data"]["modalities"].keys())
    # split into train val test
    data_train, data_test, data_val = data_class.random_train_test_split(
        test_size=config["train"]["train_split"]
    )
    # TODO: convert the above to a separate script which does this offline, otherwise
    # changing the seed will change the train val test split!!!

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
    visualize(model, train_loader, "Train sample")
    visualize(model, test_loader, "Test sample")
    visualize(model, val_loader, "Val sample")


if __name__ == "__main__":
    run_experiment()
    wandb.finish()
