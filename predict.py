from io import BytesIO
from PIL import Image
import numpy as np
from keras import models

#Species names
species_names = {
    0:"andrena hattorfiana (large scabious mining bee)",
    1:"andrena helvola",
    2:"andrena marginata (small scabious mining bee)",
    3:"andrena niveata (long-fringed mini-mining bee)",
    4:"andrena proxima (broad-faced mining bee)",
    5:"andrena rosae (perkins' mining bee)",
    6:"andrena tarsata (tormentil mining bee)",
    7:"apis mellifera",
    8:"bombus hortorum (garden bumblebee)",
    9:"bombus humilis (brown-banded carder bee)",
    10:"bombus hypnorum (tree bumblebee)",
    11:"bombus lapidarius (red-tailed bumblebee)",
    12:"bombus lucorum (white-tailed bumblebee)",
    13:"bombus muscorum (moss carder bee)",
    14:"bombus pascuorum (common carder bee)",
    15:"bombus pratorum",
    16:"bombus ruderarius (red-shanked carder bee)",
    17:"bombus ruderatus (large garden bumblebee)",
    18:"bombus soroeensis (broken-belted bumblebee)",
    19:"bombus sylvarum (shrill carder bee)",
    20:"bombus terrestris ssp. audax (buff-tailed bumblebee)",
    21:"coelioxys mandibularis (square-jawed sharp-tail bee)",
    22:"colletes cunicularius (vernal colletes bee)",
    23:"eucera longicornis (long-horned bee)",
    24:"nomad roberjeotiana (tormentil nomad bee)",
    25:"nomada argentata (silver-sided nomad bee)",
    26:"nomada fulvicornis (orange-haired nomad bee)",
    27:"nomada hirtipes (long-horned nomad bee)",
    28:"nomada signata (broad-banded nomad bee)",
    29:"osima xanthomelana (large mason bee)",
    30:"osmia bicornis",
    31:"osmia parietina (wall mason bee)",
    32:"sphecodes scabricollis (rough-backed blood bee)",
    33:"sphecodes spinulosus (spined blood bee)",
    34:"stelis ornatula (spotted dark bee)",
    35:"stelis phaeoptera (plain dark bee)"
}


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess_image(image: Image.Image):
    image = image.resize((224,224))
    image = np.asarray(image)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    return image

def load_model():
    model = models.load_model('cropped_species_classification_model_v2.h5')
    return model

def generate_prediction(model, image):
    # Make predictions using model
    preds = model.predict(image)
    return preds

def top_matches(predictions):
    top_matches = np.argsort(predictions[0])[::-1][:4]

    # Get the corresponding class labels and probabilities
    class_labels = [species_names[i] for i in top_matches]  # Replace your_class_labels with your actual class labels
    probabilities = predictions[0][top_matches]

    return class_labels, probabilities

    # # Print the top two predictions
    # for label, probability in zip(class_labels, probabilities):
    #     print(f"Class: {label}, Probability: {probability}")