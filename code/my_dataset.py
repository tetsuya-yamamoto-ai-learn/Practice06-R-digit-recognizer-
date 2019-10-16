import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        images = self.X[idx].reshape(-1, 28, 28).astype(np.float32)
        labels = self.y[idx]
        return images, labels


class TestDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):  # lenが呼ばれたときに反応する
        return len(self.X)

    def __getitem__(self, idx): # インデックスがついて呼ばれたときに反応する
        labels = self.X[idx].reshape(-1, 28, 28).astype(np.float32)
        return labels

def main():
    X_dummy = np.ones(shape=(5, 784))
    y_dummy = np.ones(shape=(5, ))

    train_dataset = TrainDataset(X_dummy, y_dummy)

    # インデックスを当てて__getitem__がうまく使用できているか確認
    image, label = train_dataset[0]
    print(image.shape)
    print(label)

    # DataLoader → データセットからバッチサイズに合わせてデータを取り出す仕組み
    train_loader = DataLoader(train_dataset, batch_size=2)
    for images, labels in train_loader:
        print(images.size(), labels.size())


if __name__ == '__main__':
    main()
