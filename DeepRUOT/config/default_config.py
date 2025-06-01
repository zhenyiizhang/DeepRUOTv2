from DeepRUOT.constants import RES_DIR
from typing import List, Optional, Union, Dict, Any

class ConfigParam:
    def __init__(self, default: Any, type: type, choices: Optional[List] = None, help: str = ""):
        self.default = default
        self.type = type
        self.choices = choices
        self.help = help

# Configuration specification in a format similar to argparse
CONFIG_SPEC = {
    # Runtime parameters
    "cuda": ConfigParam(
        default=True,
        type=bool,
        help="Whether or not to use CUDA"
    ),
    "apply_losses_in_time": ConfigParam(
        default=True,
        type=bool,
        help="Applies the losses and does back propagation as soon as a loss is calculated"
    ),
    "use_pinn": ConfigParam(
        default=True,
        type=bool,
        help="Whether to use physics-informed neural network approach"
    ),
    "use_penalty": ConfigParam(
        default=True,
        type=bool,
        help="Whether to use penalty in the loss function"
    ),
    "use_density_loss": ConfigParam(
        default=False,
        type=bool,
        help="Whether or not to add density regularization"
    ),
    "lambda_density": ConfigParam(
        default=10,
        type=float,
        help="The weight for density loss"
    ),
    "hinge_value": ConfigParam(
        default=0,
        type=float,
        help="Hinge value for density loss function"
    ),
    "top_k": ConfigParam(
        default=None,
        type=Optional[int],
        help="The k for the k-NN used in the density loss"
    ),
    "sample_with_replacement": ConfigParam(
        default=True,
        type=bool,
        help="Whether or not to sample with replacement"
    ),
    "device": ConfigParam(
        default="cuda",
        type=str,
        choices=["cuda", "cpu", 'mps'],
        help="Device to use for training"
    ),
    "sample_size": ConfigParam(
        default=1024,
        type=int,
        help="Number of points to sample during each batch"
    ),

    # Experiment parameters
    "exp": {
        "name": ConfigParam(
            default="default_experiment",
            type=str,
            help="Name of the experiment. If none is provided timestamp is used"
        ),
        "output_dir": ConfigParam(
            default=RES_DIR,
            type=str,
            help="Where experiments should be saved. The results directory will automatically be generated here"
        )
    },

    # Dataset parameters
    "data": {
        "dataset": ConfigParam(
            default="file",
            type=str,
            choices=["file"],
            help="Dataset of the experiment to use. If value is fullpath to a file then tries to load file"
        ),
        "time_col": ConfigParam(
            default=None,
            type=Optional[str],
            choices=["simulation_i", "step_ix", "sim_time"],
            help="Time column of the dataset to use"
        ),
        "file_path": ConfigParam(
            default=None,
            type=Optional[str],
            help="Path to the data file relative to DATA_DIR"
        ),
        "dim": ConfigParam(
            default=None,
            type=Optional[int],
            help="Dimension of the data"
        ),
        "hold_out": ConfigParam(
            default=None,
            type=Optional[Union[int, str]],
            help="Which time point to hold out when calculating the global loss"
        ),
        "hold_one_out": ConfigParam(
            default=True,
            type=bool,
            help="Whether or not to randomly hold one time pair out when computing the global loss"
        )
    },

    # Model architecture parameters
    "model": {
        "in_out_dim": ConfigParam(
            default=None,
            type=Optional[int],
            help="Input and output dimension of the model"
        ),
        "hidden_dim": ConfigParam(
            default=400,
            type=int,
            help="Hidden dimension size"
        ),
        "n_hiddens": ConfigParam(
            default=2,
            type=int,
            help="Number of hidden layers"
        ),
        "activation": ConfigParam(
            default="leakyrelu",
            type=str,
            choices=["Tanh", "relu", "elu", "leakyrelu"],
            help="Activation function to use"
        ),
        "score_hidden_dim": ConfigParam(
            default=128,
            type=int,
            help="Hidden dimension for score network"
        )
    },

    # Training parameters
    "training": {
        "local_epochs": ConfigParam(
            default=5,
            type=int,
            help="Number of epochs to use local_loss while training (occurs first)"
        ),
        "epochs": ConfigParam(
            default=15,
            type=int,
            help="Number of epochs to use global_loss while training"
        ),
        "local_post_epochs": ConfigParam(
            default=5,
            type=int,
            help="Number of epochs to use local_loss after training (occurs last)"
        ),
        "batches": ConfigParam(
            default=100,
            type=int,
            help="Number of batches to randomly sample from each consecutive pair of groups"
        )
    },

    # Loss parameters
    "loss": {
        "criterion": ConfigParam(
            default="ot1",
            type=str,
            choices=["mmd", "ot1"],
            help="Loss function to use"
        ),
        "lambda_local": ConfigParam(
            default=0.2,
            type=float,
            help="Weight for average local loss (lambda_local + lambda_global = 1.0)"
        ),
        "lambda_global": ConfigParam(
            default=0.8,
            type=float,
            help="Weight for global loss (lambda_local + lambda_global = 1.0)"
        ),
        "lambda_density_local": ConfigParam(
            default=1.0,
            type=float,
            help="Weight for local density loss"
        )
    },

    # Pretraining parameters
    "pretrain": {
        "epochs": ConfigParam(
            default=500,
            type=int,
            help="Number of pretraining epochs"
        ),
        "lr": ConfigParam(
            default=1e-4,
            type=float,
            help="Learning rate for pretraining"
        ),
        "lambda_ot": ConfigParam(
            default=1.0,
            type=float,
            help="Weight for optimal transport loss during pretraining"
        ),
        "lambda_mass": ConfigParam(
            default=0.01,
            type=float,
            help="Weight for mass conservation loss during pretraining"
        ),
        "lambda_energy": ConfigParam(
            default=0.0,
            type=float,
            help="Weight for energy conservation loss during pretraining"
        )
    },

    # Score training parameters
    "score_train": {
        "epochs": ConfigParam(
            default=3001,
            type=int,
            help="Number of score training epochs"
        ),
        "lr": ConfigParam(
            default=1e-4,
            type=float,
            help="Learning rate for score training"
        ),
        "lambda_penalty": ConfigParam(
            default=1,
            type=float,
            help="Weight for penalty term in score training"
        ),
        "sigma": ConfigParam(
            default=0.1,
            type=float,
            help="Sigma parameter for score training"
        )
    },

    # Main training parameters
    "train": {
        "epochs": ConfigParam(
            default=500,
            type=int,
            help="Number of main training epochs"
        ),
        "lr": ConfigParam(
            default=1e-4,
            type=float,
            help="Learning rate for main training"
        ),
        "lambda_ot": ConfigParam(
            default=10,
            type=float,
            help="Weight for optimal transport loss"
        ),
        "lambda_mass": ConfigParam(
            default=10,
            type=float,
            help="Weight for mass conservation loss"
        ),
        "lambda_energy": ConfigParam(
            default=0.01,
            type=float,
            help="Weight for energy conservation loss"
        ),
        "lambda_pinn": ConfigParam(
            default=100,
            type=float,
            help="Weight for PINN loss"
        ),
        "lambda_initial": ConfigParam(
            default=0.1,
            type=float,
            help="Weight for initial condition loss"
        ),
        "scheduler_step_size": ConfigParam(
            default=100,
            type=int,
            help="Step size for learning rate scheduler"
        ),
        "scheduler_gamma": ConfigParam(
            default=0.8,
            type=float,
            help="Gamma parameter for learning rate scheduler"
        )
    },

    # Evaluation parameters
    "eval": {
        "n_points": ConfigParam(
            default=100,
            type=int,
            help="Number of points to generate for evaluation"
        ),
        "n_trajectories": ConfigParam(
            default=30,
            type=int,
            help="Number of trajectories to generate for plot"
        ),
        "n_bins": ConfigParam(
            default=100,
            type=int,
            help="Number of bins to use for generating trajectories"
        )
    },

    # Geodesic parameters
    "geo": {
        "use_geo": ConfigParam(
            default=False,
            type=bool,
            help="Whether or not to use a geodesic embedding"
        ),
        "geo_layers": ConfigParam(
            default=[32],
            type=List[int],
            help="Layer sizes for geodesic embedding model"
        ),
        "geo_features": ConfigParam(
            default=5,
            type=int,
            help="Number of features for geodesic model"
        )
    }
}

def _extract_default_config(config_spec: Dict) -> Dict:
    """Extract default values from config specification"""
    default_config = {}
    for key, value in config_spec.items():
        if isinstance(value, dict):
            default_config[key] = _extract_default_config(value)
        else:
            default_config[key] = value.default
    return default_config

# Generate DEFAULT_CONFIG from CONFIG_SPEC
DEFAULT_CONFIG = _extract_default_config(CONFIG_SPEC)

def _extract_valid_options(config_spec: Dict, parent_key: str = "") -> Dict:
    """Extract valid options from config specification"""
    valid_options = {}
    for key, value in config_spec.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            valid_options.update(_extract_valid_options(value, current_key))
        elif value.choices is not None:
            valid_options[current_key] = value.choices
    return valid_options

# Generate VALID_OPTIONS from CONFIG_SPEC
VALID_OPTIONS = _extract_valid_options(CONFIG_SPEC) 