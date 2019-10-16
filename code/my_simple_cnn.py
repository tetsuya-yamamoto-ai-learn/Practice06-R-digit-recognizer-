import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(2, 2), stride=1)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=(2, 2), stride=1)

        # 全結合層
        self.fc1 = nn.Linear(6 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

        # プーリング層
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2))  # pooling層

        self.relu = nn.ReLU(inplace=True)  # relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = x.view(-1, 6 * 6 * 6)  # torch.tensorの配列の形を成形するメソッド(特徴マップから特徴ベクトルへ！)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def main():
    images = torch.ones(size=(5, 1, 28, 28))  # (N, C, W, H)
    # (N(データ数(batch数)), C(チャンネル数), W(横), H(縦))

    net = SimpleCNN()
    outputs = net(images)

    assert torch.Size([5, 10]) == outputs.size()


if __name__ == '__main__':
    main()
