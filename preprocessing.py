# Import the os module for file and directory operations
import os
# Import numpy for numerical operations (not used directly here, but often useful for image data)
import numpy as np
# Import ImageDataGenerator for image preprocessing and augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import pickle for saving Python objects to disk (used for saving class indices)
import pickle

# Fix paths: Ensure we're working relative to project root
SCRIPT_DIR = os.path.dirname(__file__)         # Get the directory of the current script (src/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)     # Get the parent directory (project root: "Fifth project/")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Path to the data folder containing images
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # Path to the models folder for saving files

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Set the image size to which all images will be resized (width, height)
IMG_SIZE = (224, 224)
# Set the number of images per batch for training/validation
BATCH_SIZE = 32

def create_generators(data_dir):            
    # ← raw images are transformed into a standardized, augmented, and efficiently loadable format suitable for Convolutional Neural Networks (CNNs).
    """Create train/validation generators with augmentation."""
    # Create an ImageDataGenerator object for preprocessing and augmenting images
    datagen = ImageDataGenerator(
        rescale=1./255,                     # ← Normalization: pixel values 0–1 , This normalization improves model convergence during training by ensuring consistent input scale.
        rotation_range=20,                  # ← Augmentation: random rotate images up to 20 degrees
        width_shift_range=0.2,              # ← Augmentation: randomly shift images horizontally by up to 20% of width
        height_shift_range=0.2,             # ← Augmentation: randomly shift images vertically by up to 20% of height
        horizontal_flip=True,               # ← Augmentation: randomly flip images horizontally
        zoom_range=0.2,                     # ← Augmentation: randomly zoom in/out by up to 20%
        brightness_range=[0.8, 1.2],        # ← Augmentation: randomly change image brightness between 80% and 120%
        validation_split=0.2                # ← This line does the SPLIT! This allows the model to be trained on one set and evaluated on unseen data during training to prevent overfitting.
    )

    # Create the training data generator (uses 80% of the data)
    train_gen = datagen.flow_from_directory( 
        data_dir,
        target_size=IMG_SIZE,                # ← Resize all images to 224x224 , because deep learning models require fixed-size inputs.   
        batch_size=BATCH_SIZE,
        class_mode='categorical',            # ← Use categorical labels (one-hot encoded) for multi-class classification
        subset='training',                   # ← Use the training subset (80% of data)
        shuffle=True                         # ← Shuffle images for better training
    )

    # Create the validation data generator (uses 20% of the data)
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',                 # ← Use the validation subset (20% of data)
        shuffle=False                        # ← Do not shuffle for validation (to keep order consistent)
    )

    # Save class indices (mapping from class names to integer labels) to a file for later use
    class_indices_path = os.path.join(MODELS_DIR, 'class_indices.pkl')
    with open(class_indices_path, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)
    print(f"\n✅ Class indices saved to: {class_indices_path}")

    # Return the training and validation generators
    return train_gen, val_gen

# If this script is run directly (not imported as a module), execute the following block
if __name__ == "__main__":
    # Print the path to the data directory
    print(f"Looking for data in: {DATA_DIR}")
    # Check if the data directory exists; if not, raise an error
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Create the training and validation generators
    train_gen, val_gen = create_generators(DATA_DIR)

    # Print confirmation and information about the generators
    print("\n✅ Generators created successfully!")
    print("Classes found:", train_gen.class_indices)
    print("Training batches:", len(train_gen))
    print("Validation batches:", len(val_gen))

    # To improve model generalization and robustness, several augmentation techniques are applied

    # What it means:

    # The script created two data generators:

    # train_gen: Feeds images to the model during training
    # val_gen: Tests model performance after each epoch
    
    # Training batches: 21 → Each batch has 32 images → 21 × 32 = ~672 (close to your 643; last batch is smaller)
    # Validation batches: 5 → 5 × 32 = 160 (close to 158)