# This is the only file which user should edit to get his job done.

APP_NAME = "src/object_detection_app.py"

MODEL_BUCKET_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/weights-384px-adam2%2B%2B.pth.zip"
SAVE_PATH = "data/demo_model.pth.zip"
DATA_PATH = "data/"
MODEL_PATH = "data//weights-384px-adam2++.pth"

# Some sample images over internet that you may like to give. Enter urls of images here.
SAMPLE_IMAGES = [
    "sample_images//cat0.jpg",
    "sample_images//cat1.jpg",
    "sample_images//dog0.jpg",
    "sample_images//dog1.jpg",
    "sample_images//dog2.jpg",
    "sample_images//dog3.jpg",
    "sample_images//dog4.jpg",
    "sample_images//dog5.jpg",
]

NUM_CLASSES = 38  # Hyperparameters of model
# Note this might be diffrent from len(OBJECTS_TO_DETECT) as you may have an extra background class.

# Optionally the user can simply provide classes which the model was trained on
OBJECTS_TO_DETECT = [
    "background",
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
]
