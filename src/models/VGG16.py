import torch
import torch.nn as nn
import yaml

from configs.cfg import VGG16_CONFIG_PATH


class VGG16(nn.Module):
    def __init__(self, config=None):
        super(VGG16, self).__init__()
        self.config = (
            yaml.safe_load(open(VGG16_CONFIG_PATH)) if config is None else config
        )
        self.features = nn.Sequential(
            nn.Conv2d(self.config["input_channels"], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._flattened_feature_size = self._get_flattened_feature_size(
            self.config["input_channels"],
            self.config["input_width"],
            self.config["input_height"],
        )

        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.config["num_classes"]),
        )

    def _get_flattened_feature_size(self, input_channels, input_width, input_height):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_width, input_height)
            features_out = self.features(dummy_input)
            return features_out.numel()  # Total number of elements in the output tensor

    def set_config_from_dataloader(self, dataloader):
        for images, _ in dataloader:
            _, channel, width, height = images.shape
        self.config["input_channels"] = channel
        self.config["input_width"] = width
        self.config["input_height"] = height

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
