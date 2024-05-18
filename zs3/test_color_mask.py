from PIL import Image
import numpy as np
import os

# # Convertir l'image en tableau NumPy
# img_np = np.array(img)

# print(img_np.shape)
# print(len(img_np.shape))

# # vérifier si la dimension de l'image
# if len(img_np.shape) == 2:
#     # Image en niveaux de gris
#     unique_colors = np.unique(img_np)
# else:
#     # Image en couleur
#     unique_colors = np.unique(img_np.reshape(-1, img_np.shape[2]), axis=0)

# print(unique_colors)

# Transforme l'image en noir et gris (2 couleurs pas plus)
# Permet de retirer les dégradés de gris dans les masques pour plantdoc
def binarize_image(img_np, threshold=19):
    img_np[img_np < threshold] = 0
    img_np[img_np >= threshold] = 38
    return img_np

# Binarize toutes les images en niveaux de gris dans un répertoire
def binarize_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath)
            img_np = np.array(img)
            img_np = binarize_image(img_np)
            img_bin = Image.fromarray(img_np)
            img_bin.save(filepath)

# Répertoires des masques de PlantDoc à binariser
binarize_images_in_directory('/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train/masks')
binarize_images_in_directory('/Users/alex/Documents/GitHub/ZS3/data/plantdoc/test/masks')

# Voir le pourcentage de pixels de chaque couleur dans chaque maque :
def get_color_percentage(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath)
            img_np = np.array(img)
            if len(img_np.shape) == 2:
                # Image en niveaux de gris
                unique_colors = np.unique(img_np)
            else:
                # Image en couleur
                unique_colors = np.unique(img_np.reshape(-1, img_np.shape[2]), axis=0)
            print(f"File: {filename}")
            for color in unique_colors:
                print("Color:", color, "Percentage:", np.count_nonzero(img_np == color) / img_np.size * 100)
            print()

# Répertoires des masques de PlantDoc à analyser
# Cela permet de voir si les masques sont binarisés correctement et si les classes sont bien réparties
# Ce n'est pas le cas pour les masques de PlantDoc (mal répartis)
get_color_percentage('/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train/masks')