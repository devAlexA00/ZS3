# TER sur le Zero-Shot Semantic Segmentation - ZS3Net sur Plantdoc

Ce travail reprends le code du projet ZS3 sur le modèle ZS3Net de Valeo. Le but est d'utiliser le modèle sur un jeu de données personnalisé nommé plantdoc.

## Plantdoc

Il s'agit ici d'un jeu de données composées d'images de plantes possédant éventuellement une partie malade. 

### Organisation des fichiers

```shell
ZS3/data/plantdoc
├── test
│   ├── images
│   └──  masks
└── train
    ├── images
    └── masks
```

## Modifications apportées

### Metal Performance Shaders (MPS)

Mon matériel étant un Macbook pro M2 MAX, possédant une architecture processeur ARM64, la technologie d'accélération des calculs CUDA n'est pas compatible. J'ai donc remplacé les appels à CUDA par des appels à MPS. Cela a permis d'obtenir un entraînement netement plus rapide qu'avec l'usage simple du CPU (10 fois plus rapide environ).

### Autres modifications

D'autre modifications ont été apportées et commentées afin d'adapter le code au jeu de données plantdoc. Par exemple l'ajout du fichier `train_plantdoc.py` qui gère le cas où `--dataset plantdoc` est utilisé.

## Utilisation

```Shell
python train_plantdoc.py
```

* Main options
    - `imagenet_pretrained_path`: Path to ImageNet pretrained weights.
    - `exp_path`: Path to saved logs and weights folder.
    - `checkname`: Name of the saved logs and weights folder.
    - `unseen_classes_idx`: List of idx of unseen classes.