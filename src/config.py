# This is the only file which user should edit to get his job done.

APP_NAME = "src/object_detection_app.py"

MODEL_BUCKET_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/weights-384px-adam2%2B%2B.pth.zip"
SAVE_PATH = "data/demo_model.pth.zip"
DATA_PATH = "data/"
MODEL_PATH = "data//weights-384px-adam2++.pth"

# Some sample images over internet that you may like to give. Enter urls of images here.
SAMPLE_IMAGES = ["sample_images//cat0.jpg", "sample_images//cat1.jpg", 
"sample_images//cat2.jpg", "sample_images//dog0.jpg", "sample_images//dog1.jpg"]

NUM_CLASSES = 38 # Hyperparameters of model
# Note this might be diffrent from len(OBJECTS_TO_DETECT) as you may have an extra background class.

# IMG_SIZE = (224, 224) # If you need a image size

# Optionally the user can simply provide classes which the model was trained on
OBJECTS_TO_DETECT = [
    "Abyssinian", "great_pyrenees", "Bombay", "Persian", "samoyed",
    "Maine_Coon", "havanese", "beagle", "yorkshire_terrier",
    "pomeranian", "scottish_terrier", "saint_bernard",
    "Siamese", "chihuahua", "Birman", "american_pit_bull_terrier", "miniature_pinscher",
    "japanese_chin", "British_Shorthair", "Bengal", "Russian_Blue", "newfoundland",
    "wheaten_terrier", "Ragdoll", "leonberger", "english_cocker_spaniel",
    "english_setter", "staffordshire_bull_terrier", "german_shorthaired", "Egyptian_Mau", 
    "boxer", "shiba_inu", "keeshond", "pug", "american_bulldog", "basset_hound", "Sphynx",
]


