import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/mnt/userdata/result/mnt/userdata/result/dataset/train.csv')

train_data, temp_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=data['target']
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    shuffle=True,
    stratify=temp_data['target']
)
train_data.to_csv('/mnt/userdata/result/mnt/userdata/result/dataset/train_set.csv', index=False,encoding='utf-8')
val_data.to_csv('/mnt/userdata/result/mnt/userdata/result/dataset/val_set.csv', index=False,encoding='utf-8')
test_data.to_csv('/mnt/userdata/result/mnt/userdata/result/dataset/val_set.csv', index=False,encoding='utf-8')

print(train_data.head())
print(val_data.head())
print(test_data.head())