from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.config import (
    TRAIN_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE,
    CLASSES, RANDOM_SEED,
)


def get_train_val_generators(validation_split: float = 0.2):
    """
    Return (train_gen, val_gen) from TRAIN_DIR.

    Augmentation is applied only to the training split.
    Both splits use VGG16 preprocessing (channel-wise mean subtraction).
    Classes are always loaded in the order defined by CLASSES in config.py
    so label indices are deterministic regardless of filesystem ordering.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    common = dict(
        directory=TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,       # ← forces deterministic label ordering
        seed=RANDOM_SEED,
    )

    train_gen = datagen.flow_from_directory(subset="training",  shuffle=True,  **common)
    val_gen   = datagen.flow_from_directory(subset="validation", shuffle=False, **common)
    return train_gen, val_gen


def get_test_generator():
    """
    Return a test generator (no augmentation, no shuffle).
    Classes loaded in CLASSES order for deterministic labels.
    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    return test_gen