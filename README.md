
## Download data
- [参考: Google Colab上でKaggleのデータをロード、モデル訓練、提出の全てを行う \- Qiita](https://qiita.com/katsu1110/items/a8d508a1b6f07bd3a243)

### Create New API
- https://www.kaggle.com/{YOUR_KAGGLE_ACCOUNT_NAME}/account

### Download
```bash
kaggle competitions download -c digit-recognizer
unzip digit-recognizer.zip

mkdir input
mv train.csv input/
mv test.csv input/
mv sample_submission.csv input/
```
