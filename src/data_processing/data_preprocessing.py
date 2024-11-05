import test
import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from configs.cfg import DATA_CONFIG_PATH, DATA_PATH
import yaml


class Data_Preprocessing:
    def __init__(self, dataset_name, config=None):
        self.dataset_name = dataset_name
        self.config = (
            yaml.safe_load(open(DATA_CONFIG_PATH))["datasets"][dataset_name]
            if config is None
            else config
        )

    def get_transforms(self, transform_config):
        transform_list = []
        for t in transform_config:
            if t["name"] == "RandomCrop":
                transform_list.append(
                    transforms.RandomCrop(t["size"], padding=t["padding"])
                )
            elif t["name"] == "RandomHorizontalFlip":
                transform_list.append(transforms.RandomHorizontalFlip())
            elif t["name"] == "ToTensor":
                transform_list.append(transforms.ToTensor())
            elif t["name"] == "Normalize":
                transform_list.append(
                    transforms.Normalize(mean=t["mean"], std=t["std"])
                )
        return transforms.Compose(transform_list)

    def get_dataset(self, dataset_name, train, download, transform):
        if dataset_name == "CIFAR10":
            return torchvision.datasets.CIFAR10(
                root=DATA_PATH, train=train, download=download, transform=transform
            )
        elif dataset_name == "CIFAR100":
            return torchvision.datasets.CIFAR100(
                root=DATA_PATH, train=train, download=download, transform=transform
            )
        elif dataset_name == "MNIST":
            return torchvision.datasets.MNIST(
                root=DATA_PATH, train=train, download=download, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def get_dataloader(self):
        transform_train = self.get_transforms(self.config["transforms"]["train"])
        transform_test = self.get_transforms(self.config["transforms"]["test"])

        train_dataset = self.get_dataset(
            self.dataset_name,
            train=True,
            download=self.config["download"],
            transform=transform_train,
        )
        test_dataset = self.get_dataset(
            self.dataset_name,
            train=False,
            download=self.config["download"],
            transform=transform_test,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

        return train_dataloader, test_dataloader
