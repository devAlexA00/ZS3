import os
from sklearn.model_selection import train_test_split

trainval_dir = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train/masks'
all_files = os.listdir(trainval_dir)
all_files = [f for f in all_files if f.endswith('.png')]
#print(len(all_files) != 0)

train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_txt = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train.txt'
val_txt = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/val.txt'

with open(train_txt, 'w') as f:
    for file in train_files:
        f.write(f"{file.replace('.png', '')}\n")

with open(val_txt, 'w') as f:
    for file in val_files:
        f.write(f"{file.replace('.png', '')}\n")