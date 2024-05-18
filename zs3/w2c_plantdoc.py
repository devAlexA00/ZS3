from gensim.models import Word2Vec
import numpy as np

# 1 classe non vue : "diseased plant area" non intégré dans le corpus
# Ce corpus est utilisé pour l'entraînement du modèle Word2Vec
# Les mots utilisés sont sensés être des mots-clés pour les classes "background" et "diseased plant area" pour un embedding
corpus = [
    # background
    ["background", "is", "everything", "that", "is", "not", "the", "plant"],
    ["background", "environment", "surroundings", "landscape", "setting"],
    ["natural", "habitat", "ecosystem", "wilderness", "terrain"],
    ["scenery", "view", "panorama", "outlook", "vista"],
    ["non-vegetated", "area", "barren", "terrain", "undeveloped", "land"],
    ["wild", "plants", "native", "flora", "vegetation", "species"],
    ["weathered", "rocks", "stones", "boulders", "rocky", "outcrop"],
    ["soil", "composition", "texture", "mineral", "content", "structure"],
    ["natural", "elements", "sunlight", "rainfall", "wind", "temperature"],
    ["geographic", "features", "topography", "elevation", "slope", "aspect"],
    ["seasonal", "changes", "weather", "patterns", "climate", "variations"],
    ["surrounding", "wildlife", "animals", "insects", "birds", "creatures"],
    ["human", "activities", "land", "use", "development", "urbanization"],
    ["background", "noise", "ambient", "sounds", "natural", "acoustics"],
    ["open", "space", "expansive", "area", "unobstructed", "view"],
    ["natural", "scents", "aromas", "fragrances", "from", "flora", "and", "fauna"],
    ["sky", "clouds", "weather", "patterns", "atmospheric", "conditions"],
    ["ground", "cover", "vegetation", "undergrowth", "forest", "floor"],
    ["water", "sources", "streams", "rivers", "ponds", "lakes"],
    ["fluctuations", "in", "humidity", "precipitation", "levels", "moisture", "content"],
    ["seasonal", "changes", "and", "transitions", "cycle", "of", "growth", "and", "decay"],
    # diseased plant area
    ["diseased", "plant", "area", "is", "unhealthy", "and", "struggling", "living", "area", "of", "the", "plant"],
    ["unhealthy", "plant", "struggling", "declining", "weak", "environmental", "suboptimal", "conditions"],
    ["susceptible", "vulnerable", "weak", "diseased", "plants"],
    ["weakness", "listlessness", "decline", "wither", "deteriorate"],
    ["unbalanced", "nutrient", "poor", "infertile", "soil", "diseased", "microbes"],
    ["contaminated", "infected", "diseased", "area", "zone"],
    ["virus-ridden", "disease-ridden", "pest-ridden", "diseased", "plants"],
    ["weak", "root", "system", "diseased", "nematodes"],
    ["diseased", "protozoa", "harmful", "microorganisms"],
    ["impaired", "physiological", "plant", "functions", "poor", "growth"],
    ["outbreak", "epidemic", "diseased", "zone"],
    ["non-organic", "chemical", "farming", "practices", "diseased", "crops"],
    ["biological", "imbalance", "ecosystem", "diseased", "fields"],
    ["struggling", "farmers", "worldwide", "crops", "species", "low", "yields"],
    ["sparse", "foliage", "few", "leaves", "unhealthy", "greenery"],
    ["weak", "stem", "diseased", "branches", "fragile", "structure"],
    ["few", "blossoms", "faded", "flowers", "wilting", "decay"],
    ["meager", "harvest", "scarce", "yield", "unproductive", "crop"],
    ["feeble", "photosynthesis", "weak", "chlorophyll", "slow", "metabolism"],
    ["disrupted", "ecosystem", "imbalanced", "biodiversity", "diseased", "interactions"],
    ["inadequate", "water", "supply", "dehydrated", "moisture", "loss"],
    ["poor", "light", "exposure", "inadequate", "sunshine", "photosynthesis"],
    ["sparse", "vegetation", "thin", "green", "canopy", "struggling", "habitat"],
    ["weak", "pollinator", "presence", "poor", "reproduction", "depleted", "ecosystem"],
    ["fragile", "defense", "mechanisms", "susceptible", "to", "pests", "and", "diseases"],
    ["unbalanced", "nutrient", "uptake", "deficient", "minerals", "and", "vitamins"],
    ["negligent", "monitoring", "and", "maintenance", "late", "detection", "of", "issues"]
]

# Entraîner le modèle Word2Vec
model = Word2Vec(corpus, vector_size=300, window=5, min_count=1, workers=4)
embeddings = {word: model.wv[word] for word in model.wv.key_to_index}

# Sauvegarder les embeddings
np.save('/Users/alex/Documents/GitHub/ZS3/zs3/embeddings/plantdoc/plantdoc_class_w2c.npy', embeddings)
