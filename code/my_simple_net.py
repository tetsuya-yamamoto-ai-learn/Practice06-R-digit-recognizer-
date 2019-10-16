import torch
from torch import nn


class MySimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=196)
        self.fc2 = nn.Linear(in_features=196, out_features=49)
        self.fc3 = nn.Linear(in_features=49, out_features=10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    model = MySimpleNet()

    data_num = 5
    dummy_inputs = torch.ones(size=(data_num, 784))

    outputs = model(dummy_inputs)

    assert torch.Size([data_num, 10]) == outputs.size()


if __name__ == '__main__':
    main()
