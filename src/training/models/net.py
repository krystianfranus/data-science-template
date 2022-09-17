from torch import nn


class LinearRegressionNet(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x).flatten()
