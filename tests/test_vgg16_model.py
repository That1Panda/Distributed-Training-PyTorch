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
            sample["batch_size"],
            sample["channels"],
            sample["height"],
            sample["width"],
            sample["num_classes"],
            sample["response"],
        )


class TestVGG16:
    @pytest.mark.parametrize(
        "batch_size, channels, height, width, num_classes, response", load_test_data()
    )
    def test_vgg16_model(
        self, batch_size, channels, height, width, num_classes, response
    ):
        model = VGG16(channels, width, height, num_classes)
        x = torch.randn(batch_size, channels, height, width)
        y = model(x)
        assert y.size()[0] == batch_size
        assert y.size()[1] == num_classes
        assert len(y.size()) == len(response)
