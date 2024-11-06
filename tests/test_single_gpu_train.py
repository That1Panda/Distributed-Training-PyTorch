import json
from numpy import single
import pytest

from configs.cfg import TEST_DATA_DIR
from src.training.single_gpu_train import SingleGPUTrain
import torch

DATA_FILE = TEST_DATA_DIR / "single_gpu_train.json"


def load_test_data():
    with open(DATA_FILE) as f:
        test_data = json.load(f)

    for sample in test_data:
        yield (sample["config"], sample["response"])


class TestSingleGPUTrain:
    @pytest.mark.parametrize("config, response", load_test_data())
    def test_single_gpu_train(self, config, response):
        if config is not None:
            assert "model" in config.keys()
            assert "training" in config.keys()
            assert "data" in config.keys()

        single_gpu_train = SingleGPUTrain(config)

        assert single_gpu_train.config == config
        assert single_gpu_train.device == torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        single_gpu_train.model_training()

        assert single_gpu_train.time_of_training >= response["time_of_training"]
        assert single_gpu_train.train_accuracy >= response["train_accuracy"]
        assert single_gpu_train.val_accuracy >= response["val_accuracy"]
