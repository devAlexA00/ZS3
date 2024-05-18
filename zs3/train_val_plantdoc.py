import os
from sklearn.model_selection import train_test_split

trainval_dir = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train/masks'

# Obtenir la liste de tous les fichiers dans le répertoire trainval_dir
all_files = os.listdir(trainval_dir)

# Filtrer les fichiers pour ne garder que ceux qui se terminent par '.png'
all_files = [f for f in all_files if f.endswith('.png')]

# Diviser les fichiers en ensembles d'entraînement et de validation
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Chemin d'accès au fichier train.txt
train_txt = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train.txt'

# Chemin d'accès au fichier val.txt
val_txt = '/Users/alex/Documents/GitHub/ZS3/data/plantdoc/val.txt'

# Écrire les noms des fichiers d'entraînement dans le fichier train.txt
with open(train_txt, 'w') as f:
    for file in train_files:
        f.write(f"{file.replace('.png', '')}\n")

# Écrire les noms des fichiers de validation dans le fichier val.txt
with open(val_txt, 'w') as f:
    for file in val_files:
        f.write(f"{file.replace('.png', '')}\n")