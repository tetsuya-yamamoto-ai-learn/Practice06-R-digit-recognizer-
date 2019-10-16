import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.data_path import DataPath
from code.my_dataset import TrainDataset, TestDataset
from code.my_simple_net import MySimpleNet
from code.my_simple_cnn import SimpleCNN


def prepare_data_loaders(batch_size):
    train = pd.read_csv(DataPath.TrainCsv.value)

    X = train.iloc[:, 1:].values
    y = train.iloc[:, 0].values
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          train_size=0.8,
                                                          random_state=0)
    train_dataset = TrainDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TrainDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def run_train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        # 勾配初期化
        optimizer.zero_grad()

        # 順伝播計算
        outputs = model(images)

        # 損失の計算
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # 重みの更新
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch} Train Loss: {epoch_loss:.4f}')


def run_valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    print(f'Epoch: {epoch} Valid Loss: {epoch_loss:.4f}')


def make_predictions(model, test_loader):
    model.eval()
    predictions = np.array([], dtype=np.int)

    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)

            _, y_pred = torch.max(outputs, dim=1)
            y_pred_label = y_pred.numpy()

            predictions = np.append(predictions, y_pred_label)

    return predictions


def make_submission_file(model, predictions):
    submit_data = pd.read_csv(DataPath.SubmissionCsv.value)
    submit_data['Label'] = predictions

    yymmddhhmmss = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = model.__class__.__name__

    # ex: '20201231_174530_SimpleNet.csv
    save_submission_path = DataPath.Submission.value / f'{yymmddhhmmss}_{model_name}.csv'

    submit_data.to_csv(save_submission_path, index=False)
    print(f'Saved {save_submission_path}')


def main():
    torch.manual_seed(0)

    BATCH_SIZE = 100

    # 1. DatasetとDataLoader
    train_loader, valid_loader = prepare_data_loaders(BATCH_SIZE)

    # 2. モデル(ネットワーク)
    model: nn.Module = SimpleCNN()

    # 最適化アルゴリズムと損失関数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 3. 学習
    NUM_EPOCHS = 10

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        run_train_epoch(model, train_loader, criterion, optimizer, epoch)
        run_valid_epoch(model, valid_loader, criterion, epoch)

    # 4. TestDataでの予測
    df_test = pd.read_csv(DataPath.TestCsv.value)
    X_test = df_test.values

    test_dataset = TestDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = make_predictions(model, test_loader)

    # submissionの作成
    make_submission_file(model, predictions)


if __name__ == '__main__':
    main()
