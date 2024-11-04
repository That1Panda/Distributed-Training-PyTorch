from pathlib import Path
from re import M

from yaml import safe_load

PROJECT_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_DIR / "tests"
CONFIGS_DIR = PROJECT_DIR / "configs"
VGG16_CONFIG_PATH = CONFIGS_DIR / "vgg16_config.yaml"
