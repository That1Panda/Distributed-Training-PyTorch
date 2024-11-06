import json
import pytest

from configs.cfg import TEST_DATA_DIR
from src.training.data_parallel_train import DataParallelTrain

DATA_FILE = TEST_DATA_DIR / "data_parallel_train.json"


def load_test_data():
    with open(DATA_FILE) as f:
        test_data = json.load(f)

    for sample in test_data:
        yield (sample["config"], sample["response"])


class TestDataParallelTrain:
    @pytest.mark.parametrize("config, response", load_test_data())
    def test_data_parallel_train(self, config, response):
        if config is not None:
            assert "model" in config.keys()
            assert "training" in config.keys()
            assert "data" in config.keys()

        data_parallel_train = DataParallelTrain(config)

        data_parallel_train.model_training()

        assert data_parallel_train.time_of_training >= response["time_of_training"]
        assert data_parallel_train.train_accuracy >= response["train_accuracy"]
        assert data_parallel_train.val_accuracy >= response["val_accuracy"]
