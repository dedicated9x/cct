# common constants
import torch

from pathlib  import Path

# DATA_DIR = "data"
DATA_DIR = str(Path(__file__).parents[1] / "data")
# TODO wrocic z tym do normalnosci
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
