from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Dict

from os import path

from accelerate import Accelerator
from colorama import Fore, Style

import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer

from model import MCformer


class BaseExperiment(ABC):
    """Base class for experiments."""

    def __init__(self, configs, accelerator: Accelerator) -> None:
        super().__init__()
        self.configs = configs
        self.accelerator = accelerator

        # Build the deep learning model
        self.model_name = configs.model

        # The mode of the experiment: "supervised" or "pre-training"
        self.mode = configs.mode

        # The hyper-parameters for data loading
        self.batch_size = configs.batch_size
        self.shuffle = configs.shuffle

        # The checkpoint directory
        self.checkpoint_dir = configs.checkpoint
        # self.checkpoint_path = path.join(self.checkpoint_dir, setting)
        # os.makedirs(self.checkpoint_path, exist_ok=True)

    @abstractmethod
    def run(self, setting: str) -> None:
        """Run the experiment."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def model_dict(self) -> Dict[str, nn.Module]:
        """Return a dictionary of available models."""
        return {
            "MCformer": MCformer.Model,
            # Add other models here as needed
        }

    def build_model(self, name: str = "SpectrumTime") -> nn.Module:
        """Build the model for training."""
        # Check if the model name is valid
        assert name in self.model_dict, f"Model {name} is not supported."

        self.accelerator.print(f"Building the model: {name}", end=" -> ")

        # Create the model for experiment
        model = self.model_dict[name].Model(self.configs)
        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)

        return model

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any, Any]:
        """Load the dataset for training, validation, and testing."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Obtain trainable parameters of the model in experiment."""
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Obtain the number of trainable parameters"""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_learning_rate(self, optimizer: Optimizer) -> float:
        """Get the current learning rate of the optimizer"""
        return optimizer.param_groups[0]["lr"]
