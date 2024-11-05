import json
import pytest

from configs.cfg import TEST_DATA_DIR
from src.data_processing.data_preprocessing import Data_Preprocessing
import torch

DATA_FILE = TEST_DATA_DIR / "data_preprocessing.json"


def load_test_data():
    with open(DATA_FILE) as f:
        test_data = json.load(f)

    for sample in test_data:
        yield (
            sample["dataset_name"],
            sample["config"],
            sample["response"],
        )


class TestDataPreprocessing:
    @pytest.mark.parametrize("dataset_name, config, response", load_test_data())
    def test_data_preprocessing(self, dataset_name, config, response):
        data_preprocessing = Data_Preprocessing(dataset_name, config)
        train_data, test_data = data_preprocessing.get_dataloader()

        assert len(train_data) == response["train_length"]
        assert len(test_data) == response["test_length"]
