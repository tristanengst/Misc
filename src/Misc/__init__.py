import torch
import os

################################################################################
# Set up seeds and CUDA. This has to be done first I think.
################################################################################
# Set up CUDA usage
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Turn off WandB console logging, since we don't need it and it breaks TQDM.

# Import remaining code
from .DatasetUtils import *
from .DebuggingUtils import *
from .IOUtils import *
from .LMDBUtils import *
from .LPIPSUtils import *
from .MiscUtils import *
from .NNUtils import *
from .TarImageFolder import *
from .TensorManipulation import *
