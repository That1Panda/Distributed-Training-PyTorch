from pathlib import Path
from re import M

from yaml import safe_load

PROJECT_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_DIR / "tests" / "data"
DATA_PATH = PROJECT_DIR / "data"
CONFIGS_DIR = PROJECT_DIR / "configs"
LOG_DIR = PROJECT_DIR / "logs"

VGG16_CONFIG_PATH = CONFIGS_DIR / "vgg16_config.yaml"
DATA_CONFIG_PATH = CONFIGS_DIR / "data_config.yaml"
SINGLE_GPU_TRAIN_CONFIG_PATH = CONFIGS_DIR / "single_gpu_train_config.yaml"
DATA_PARALLEL_CONFIG_PATH = CONFIGS_DIR / "data_parallel_train_config.yaml"
