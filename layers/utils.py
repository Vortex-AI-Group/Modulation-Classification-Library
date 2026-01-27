from typing import Optional, Union

import torch
from torch import nn


class SwiGLU(nn.Module):
    """The Gated Linear Unit with the Swish Function"""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None) -> None:
        """
        :param input_dim: (int) the input dimension for the linear proj in ``SwiGLU`` function.
        :param hidden_dim: Optional(int) the hidden dimension for the linear proj in ``SwiGLU`` function.
        """
        super(SwiGLU, self).__init__()

        # Determine the dimension of the hidden layer
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        # Create two linear transformations
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)

        # Using built-in Swish functions
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation part of the SwiGLU activation function.

        :param x: (Tensor) the output tensor to be activated.

        :return: (Tensor) the results of the activation.
        """
        return self.fc1(x) * self.swish(self.fc2(x))


class Activation(nn.Module):
    """
    Get the activation function to use according to the specified name.
    """

    def __init__(
        self,
        activation: Union[nn.Module, str] = "gelu",
        inplace: Optional[bool] = False,
        approximate: Optional[str] = "none",
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """
        :param activation: Union(nn.Module, str) the name of the activation function. Default: ``relu``.
        :param inplace: Optional(str) can optionally do the operation in-place. Default: ``False``
        :param approximate: Optional(str) the gelu approximation algorithm to use: ``None`` | ``tanH``.
        :param input_dim: Optional(int) the input dimension for the linear proj in ``SwiGLU`` function.
        :param hidden_dim: Optional(int) the hidden dimension for the linear proj in ``SwiGLU`` function.
        """
        super(Activation, self).__init__()

        # Determine whether the input is a directly callable object
        if callable(activation):
            self.activation = activation()

        # Determine whether it is a string
        assert isinstance(activation, str)
        self.name = activation.lower()

        # Select the activation function to use based on its name
        if self.name == "relu":
            self.activation = nn.ReLU(inplace=inplace)
        elif self.name == "gelu":
            self.activation = nn.GELU(approximate=approximate)
        elif self.name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.name == "logsigmoid":
            self.activation = nn.LogSigmoid()
        elif self.name == "swish":
            self.activation = nn.SiLU(inplace=inplace)
        elif self.name == "swiglu":
            self.activation = SwiGLU(input_dim, hidden_dim)
        else:
            raise ValueError("Please enter the correct activation function name!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation part of the activation function.

        :param x: (Tensor) the output tensor to be activated.

        :return: (Tensor) the results of the activation.
        """
        return self.activation(x)
