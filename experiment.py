import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from config.config import load_config
from src.data.datamodule import DyanmicsDataset
from src.data.datasplit import DataSplitUtils
from src.trainer.simple_trainer import TrainerModule
import os

def visualize(model, loader, title: str):
    # sample random batch from loader
    batch = next(iter(loader))
    state = batch["state"].float()
    actions = batch["action_horizon"].float()
    targets = batch["ground_truth"].float()
    traversability_cost = batch["traversability_cost"].float()
    wheel_rpm = batch["wheel_rpm"].float()
    traversability_breakdown = batch["traversability_breakdown"].float()
    height_map = batch["height_map_12x12"].float()
    rgb_map = batch["image_left_color"].float()

    predictions = model.forward(
        state,
        actions,
        traversability_cost,
        traversability_breakdown,
        wheel_rpm,
        height_map,
        rgb_map,
    )
    # plot random sample from batch
    idx = np.random.randint(0, state.shape[0])
    target_xyz = targets[idx, :, :3].detach().cpu().numpy()
    pred_xyz = predictions[idx, :, :3].detach().cpu().numpy()
    print("hi")
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
    # plt.savefig(f"images/{title}.png")
    


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/debug.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    pl.seed_everything(config["seed"])
    # Read dataset
    data_class = DataSplitUtils(config["data"]["dataset_path"])
    # _ = DataSplitUtils.load_data(
    #     data_class, config["data"]["modalities"].keys()
    # )
    # split into train val test
    # data_class.random_train_test_split(config["train"]["train_split"], config["data"]["dataset_split_path"])
    # data_class.rebalance_and_filter_split(
    #     path=config["data"]["dataset_split_path"],
    #     modalities=list(config["data"]["modalities"].keys()),
    # )
    # exit(0)

    data_train, data_test, data_val = data_class.load_random_train_test_split(
        config["data"]["dataset_split_path"]
    )

    train_dataset = DyanmicsDataset(config, data_train, "train")
    test_dataset = DyanmicsDataset(
        config,
        data_test,
        "test",
        normalization_stats=train_dataset.normalization_stats,
    )
    val_dataset = DyanmicsDataset(
        config,
        data_val,
        "val",
        normalization_stats=train_dataset.normalization_stats,
    )
    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    print(f"{len(val_dataset)=}")
    train_loader = train_dataset.get_dataloader()
    test_loader = test_dataset.get_dataloader()
    val_loader = val_dataset.get_dataloader()
    model = TrainerModule(
        config=config,
        state_size=16,
        action_size=2,
        cost_size=1,
        breakdown_size=8,
        rpm_size=4,
    )  # remove hardcoding, wrap this in a class
    wandb_logger = WandbLogger(project="229-project")
    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epochs"],
        logger=wandb_logger,
        accelerator="gpu",
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=1,
        val_check_interval=config["train"]["val_check_interval"],
        # devices=2, 
        # strategy="ddp",
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    # visualize(model, train_loader, "Train sample")
    # visualize(model, test_loader, "Test sample")
    # visualize(model, val_loader, "Val sample")


if __name__ == "__main__":
    run_experiment()
    wandb.finish()
