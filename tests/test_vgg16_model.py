import json
import pytest

from configs.cfg import TEST_DATA_DIR
from src.models.VGG16 import VGG16
import torch

DATA_FILE = TEST_DATA_DIR / "data" / "vgg16_model.json"


def load_test_data():
    with open(DATA_FILE) as f:
        test_data = json.load(f)

    for sample in test_data:
        yield (
            sample["config"],
            sample["response"],
        )


class TestVGG16:
    @pytest.mark.parametrize("config, response", load_test_data())
    def test_vgg16_model(self, config, response):
        model = VGG16(config)
        x = torch.randn(
            config["batch_size"],
            config["input_channels"],
            config["input_height"],
            config["input_width"],
        )
        y = model(x)
        assert y.size()[0] == config["batch_size"]
        assert y.size()[1] == config["num_classes"]
        assert len(y.size()) == len(response)
