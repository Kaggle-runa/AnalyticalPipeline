import datetime

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now = datetime.datetime.now(JST)

# 入力データ
dataset_path = "./data/raw/dataset.csv"

# 保存先のディレクトリ
result_dir = "./data/result"
preprocessed_dir = f"{result_dir}/preprocess"
model_dir = f"{result_dir}/model"

# ファイル名のprefix/surfix
version = "ver1"
date_jst = now.strftime("%Y%m%d%H%M%S")