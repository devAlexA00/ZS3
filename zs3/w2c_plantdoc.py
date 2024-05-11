from gensim.models import Word2Vec
import numpy as np

# 1 classe non vue : "diseased plant area" non intégré dans le corpus
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
    # healthy plant area
    ["healthy", "plant", "area", "is", "green", "and", "vibrant", "living", "area", "of", "the", "plant"],
    ["health", "plant", "thriving", "flourishing", "vigorous", "environmental", "optimal", "conditions"],
    ["resistant", "immune", "strong", "robust", "healthy", "plants"],
    ["vigor", "liveliness", "flourish", "grow", "develop", "thrive"],
    ["balanced", "nutrient", "rich", "fertile", "soil", "healthy", "microbes"],
    ["pristine", "clean", "pure", "uninfected", "area", "zone"],
    ["virus-free", "disease-free", "pest-free", "healthy", "plants"],
    ["strong", "root", "system", "healthy", "nematodes"],
    ["healthy", "protozoa", "beneficial", "microorganisms"],
    ["optimal", "physiological", "plant", "functions", "healthy", "growth"],
    ["no", "outbreak", "no", "epidemic", "healthy", "zone"],
    ["organic", "natural", "farming", "practices", "healthy", "crops"],
    ["biological", "balance", "ecosystem", "healthy", "fields"],
    ["thriving", "farmers", "worldwide", "crops", "species", "high", "yields"],
    ["lush", "foliage", "abundant", "leaves", "luxuriant", "greenery"],
    ["robust", "stem", "healthy", "branches", "sturdy", "structure"],
    ["abundant", "blossoms", "vibrant", "flowers", "blooming", "beauty"],
    ["fruitful", "harvest", "bountiful", "yield", "productive", "crop"],
    ["vigorous", "photosynthesis", "strong", "chlorophyll", "active", "metabolism"],
    ["harmonious", "ecosystem", "balanced", "biodiversity", "healthy", "interactions"],
    ["adequate", "water", "supply", "well-hydrated", "moisture", "retention"],
    ["optimal", "light", "exposure", "adequate", "sunshine", "photosynthesis"],
    ["lush", "vegetation", "dense", "green", "canopy", "flourishing", "habitat"],
    ["strong", "pollinator", "presence", "healthy", "reproduction", "diverse", "ecosystem"],
    ["robust", "defense", "mechanisms", "resilient", "to", "pests", "and", "diseases"],
    ["balanced", "nutrient", "uptake", "adequate", "minerals", "and", "vitamins"],
    ["vigilant", "monitoring", "and", "maintenance", "early", "detection", "of", "issues"]
]

model = Word2Vec(corpus, vector_size=300, window=5, min_count=1, workers=4)
embeddings = {word: model.wv[word] for word in model.wv.key_to_index}

np.save('/Users/alex/Documents/GitHub/ZS3/zs3/embeddings/plantdoc/plantdoc_class_w2c.npy', embeddings)
