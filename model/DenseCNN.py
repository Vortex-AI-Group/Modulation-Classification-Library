import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    """Bottleneck"""

    def __init__(
        self, d_model: int, growth_rate: int, bn_size: int, dropout: float
    ) -> None:
        super(_DenseLayer, self).__init__()
        # BN -> ReLU -> Conv
        self.layer = nn.Sequential(
            OrderedDict(
                [
                    ("norm1", nn.BatchNorm2d(d_model)),
                    ("relu1", nn.ReLU(inplace=True)),
                    (
                        "conv1",
                        nn.Conv2d(
                            d_model,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    ("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
                    ("relu2", nn.ReLU(inplace=True)),
                    (
                        "conv2",
                        nn.Conv2d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        new_features = self.layer(x)
        if self.dropout > 0:
            new_features = F.dropout(
                new_features, p=self.dropout, training=self.training
            )
        # Concatenate along channel dimension
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """由多个 _DenseLayer 组成的密集块"""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout: float,
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, dropout
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    """
    过渡层：用于下采样和通道压缩
    BN + ReLU + 1x1 Conv + 2x2 AvgPooling
    """

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class model(nn.Module):
    """`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_ backbone
    The input for DenseNet is a 3*H*W image (e.g., ImageNet 224x224)
    Args:
        growth_rate (int): how many filters to add each layer (k)
        block_config (list): how many layers in each of the 4 dense blocks
        d_model (int): the number of filters in the first conv layer
        bn_size (int): multiplicative factor for bottleneck layers (default 4)
        dropout (float): dropout rate after each dense layer
        n_classes (int): number of classes for classification
    """

    def __init__(
        self,
        configs,
    ) -> None:
        super(model, self).__init__()

        self.growth_rate = configs.growth_rate
        self.block_config = configs.block_config
        self.d_model = configs.d_model
        self.bn_size = configs.bn_size
        self.dropout = configs.dropout
        self.n_classes = configs.n_classes
        self.reduction = getattr(configs, "reduction", 0.5)

        # Initial Layer
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            self.d_model,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(self.d_model)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # 构建密集块 (Dense Blocks) 与过渡层 (Transition Layers)
        num_features = self.d_model
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                dropout=self.dropout,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate

            # 若不是最后一个 Block，则添加 Transition 层
            if i != len(self.block_config) - 1:
                num_output_features = int(num_features * self.reduction)
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_output_features,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_output_features

        # (Final Batch Norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, self.n_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc: torch.FloatTensor) -> torch.FloatTensor:
        features = self.features(x_enc)

        # Global Average Pooling
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return self.classifier(out)
