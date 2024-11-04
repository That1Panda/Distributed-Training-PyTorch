from pathlib import Path
from re import M

from yaml import safe_load

PROJECT_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_DIR / "tests"

# Model input parameters
MODEL_INPUT_CHANNELS = 3
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_NUM_CLASSES = 10
