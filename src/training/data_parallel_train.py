import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.models.VGG16 import VGG16
from src.data_processing.data_preprocessing import Data_Preprocessing
from configs.cfg import DATA_PARALLEL_CONFIG_PATH, LOG_DIR
import yaml
import time


class DataParallelTrain:
    def __init__(self, config=None):
        self.config = (
            yaml.safe_load(open(DATA_PARALLEL_CONFIG_PATH))["config"]
            if config is None
            else config
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_of_training = 0
        self.train_accuracy = 0
        self.val_accuracy = 0

    def train(self, model, train_data, criterion, optimizer, epoch, writer):
        model.train()
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

            if (batch_idx + 1) % self.config["training"]["log_interval"] == 0:
                print(
                    f"Train Epoch: {epoch+1} [{batch_idx*len(inputs)}/{len(train_data.dataset)}] "
                    f"Loss: {loss.item():.3f} Accuracy: {100.*correct/total:.3f}"
                )

            if self.device.type == "cpu":
                break  # For CI/CD only run one batch

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
            print(
                f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n"
            )

            writer.add_scalar("validation_loss", val_loss, epoch)
            writer.add_scalar("validation_accuracy", accuracy, epoch)

            return accuracy

    def model_training(self):
        data = Data_Preprocessing(self.config["data"]["dataset_name"])
        train_data, test_data = data.get_dataloader()
        print("Data Preprocessing Done")

        model = (
            VGG16(data.get_model_config())
            if self.config["model"]["type"] == "VGG16"
            else None
        )
        model = nn.DataParallel(model)
        model = model.to(self.device)
        print("Model Created with Data Parallel")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["training"]["lr"])
        writer = SummaryWriter(log_dir=LOG_DIR / "data_parallel_train")
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

        print(f"Training Time: {self.time_of_training}")
        print(f"Final Training Accuracy: {self.train_accuracy}")
        print(f"Final Validation Accuracy: {self.val_accuracy}")
        writer.close()
