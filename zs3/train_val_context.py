import os
from sklearn.model_selection import train_test_split

trainval_dir = '/Users/alex/Documents/GitHub/ZS3/data/context/full_annotations/trainval'
all_files = os.listdir(trainval_dir)
all_files = [f for f in all_files if f.endswith('.mat')]
#print(len(all_files) != 0)

train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_txt = '/Users/alex/Documents/GitHub/ZS3/data/context/train.txt'
val_txt = '/Users/alex/Documents/GitHub/ZS3/data/context/val.txt'

with open(train_txt, 'w') as f:
    for file in train_files:
        f.write(f"{file.replace('.mat', '')}\n")

with open(val_txt, 'w') as f:
    for file in val_files:
        f.write(f"{file.replace('.mat', '')}\n")