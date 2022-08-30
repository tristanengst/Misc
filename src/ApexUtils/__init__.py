import torch
import os

################################################################################
# Set up seeds and CUDA. This has to be done first I think.
################################################################################
# Set up CUDA usage
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# Turn off WandB console logging, since we don't need it and it breaks TQDM.
os.environ["WANDB_CONSOLE"] = "off"

# Import remaining code
from .DebuggingUtils import *
from .IOUtils import *
from .LMDBUtils import *
from .LPIPSUtils import *
from .NNUtils import *
from .TensorManipulation import *
