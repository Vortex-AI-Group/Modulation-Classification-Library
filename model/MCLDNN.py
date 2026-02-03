import torch
import torch.nn as nn


class Model(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length: int = 128, num_classes: int = -1) -> None:
        super(Model, self).__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1,
                50,
                kernel_size=(2, 8),
                padding="same",
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="same"),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding="valid"),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=100, hidden_size=128, batch_first=True, num_layers=2
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x_enc: torch.FloatTensor) -> torch.FloatTensor:

        x1 = self.conv1(x_enc)
        x2 = self.conv2(x_enc[:, :, 0, :])
        x3 = self.conv3(x_enc[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x = torch.reshape(x5, [-1, self.frame_length - 4, 100])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return x
