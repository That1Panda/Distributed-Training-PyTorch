import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from src.models.VGG16 import VGG16
from src.data_processing.data_preprocessing import Data_Preprocessing
from configs.cfg import DDP_CONFIG_PATH, LOG_DIR
import yaml
import time


class DDPTrain:
    def __init__(self, rank, config=None):
        self.rank = rank
        self.config = (
            yaml.safe_load(open(DDP_CONFIG_PATH))["config"]
            if config is None
            else config
        )
        self.device = torch.device(f"cuda:{self.rank}")
        self.time_of_training = 0
        self.train_accuracy = 0
        self.val_accuracy = 0

        self.setup()

    def setup(self):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.rank)

    def cleanup(self):
        dist.destroy_process_group()

    def train(self, model, train_data, criterion, optimizer, epoch, writer):
        model.train()
        train_data.sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (
                self.rank == 0
                and (batch_idx + 1) % self.config["training"]["log_interval"] == 0
            ):
                print(
                    f"Train Epoch: {epoch+1} [{batch_idx*len(inputs)}/{len(train_data.dataset)}] "
                    f"Loss: {loss.item():.3f} Accuracy: {100.*correct/total:.3f}"
                )

            if self.device.type == "cpu":
                break  # For CI/CD only run one batch
        if self.rank == 0:
            writer.add_scalar("training_loss", running_loss / len(train_data), epoch)
            writer.add_scalar("training_accuracy", 100.0 * correct / total, epoch)

        return 100.0 * correct / total

    def validate(self, model, test_data, criterion, epoch, writer):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if self.device.type == "cpu":
                    break  # For CI/CD only run one batch

            val_loss = running_loss / len(test_data)
            accuracy = 100.0 * correct / total

            if self.rank == 0:
                print(
                    f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
                )

                writer.add_scalar("validation_loss", val_loss, epoch)
                writer.add_scalar("validation_accuracy", accuracy, epoch)

            return accuracy

    def model_training(self):
        data = Data_Preprocessing(self.config["data"]["dataset_name"])
        train_data, test_data = data.get_dataloader()
        train_data = DataLoader(
            train_data.dataset,
            batch_size=train_data.batch_size,
            sampler=DistributedSampler(train_data.dataset),
        )
        print("Data Preprocessing Done")

        model = (
            VGG16(data.get_model_config())
            if self.config["model"]["type"] == "VGG16"
            else None
        )
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.rank])
        print("Model Created with Data Parallel")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["training"]["lr"])
        writer = (
            SummaryWriter(log_dir=LOG_DIR / "data_parallel_train")
            if self.rank == 0
            else None
        )

        print("Training Started")
        start_time = time.time()

        for epoch in range(self.config["training"]["epochs"]):
            train_acc = self.train(
                model, train_data, criterion, optimizer, epoch, writer
            )
            val_acc = self.validate(model, test_data, criterion, epoch, writer)

        self.time_of_training = time.time() - start_time
        self.train_accuracy = train_acc
        self.val_accuracy = val_acc

        if self.rank == 0:
            print(f"Training Time: {self.time_of_training}")
            print(f"Final Training Accuracy: {self.train_accuracy}")
            print(f"Final Validation Accuracy: {self.val_accuracy}")
            writer.close()

        self.cleanup()
