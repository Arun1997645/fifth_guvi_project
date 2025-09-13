# src/model_training.py
"""
SolarGuard: Model Training Script
Trains a CNN using MobileNetV2 for solar panel defect classification.
Fixed path handling and import issues.
"""

# Import the os module for environment variable and file operations
import os
# Suppress oneDNN warnings (optional, helps reduce TensorFlow log clutter)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to Python path so that imports from src/ work correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import TensorFlow and Keras modules for building and training the model
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2           # Pre-trained MobileNetV2 model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Layers for custom classifier head
from tensorflow.keras.models import Model                       # Model class for building the network
from tensorflow.keras.optimizers import Adam                    # Adam optimizer for training
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Callbacks for training control

# Import the data generator creation function from preprocessing script
from src.preprocessing import create_generators

def build_model(num_classes=6):
    """Build model using MobileNetV2."""
    # Define input shape for the model (224x224 pixels, 3 color channels)
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Load MobileNetV2 with pre-trained weights (on ImageNet), exclude top layers
    base_model = MobileNetV2(
        weights='imagenet',         # Use weights trained on ImageNet dataset
        include_top=False,          # Do not include the final classification layer
        input_tensor=inputs         # Use our custom input layer
    )

    # Freeze the base model so its weights are not updated during training
    base_model.trainable = False

    # Add custom classifier head on top of MobileNetV2
    x = GlobalAveragePooling2D()(base_model.output)  # Pool features to a single vector
    x = Dense(128, activation='relu')(x)             # Add a dense layer with ReLU activation
    outputs = Dense(num_classes, activation='softmax')(x)  # Output layer for multi-class classification

    # Create the final model object
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']        # How often the model was correct on the training data
    )
    return model

def train_model():
    """Train the model."""
    # Define the path to the data directory containing images
    DATA_DIR = r"C:\Users\ARURAVI\PM6\AIML GUVI Projects\Fifth project\data"

    # Create training and validation data generators
    print("🔄 Creating data generators...")
    train_gen, val_gen = create_generators(DATA_DIR)

    # Build the CNN model using MobileNetV2 as the base
    print("🧠 Building model using MobileNetV2...")
    model = build_model(num_classes=len(train_gen.class_indices))  # Number of classes from data

    # Print a summary of the model architecture
    model.summary()

    # Define paths for saving the trained model and training history
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)  # Create the models directory if it doesn't exist

    model_save_path = MODELS_DIR / "best_cnn_model.h5"         # Path to save the best model
    history_save_path = MODELS_DIR / "training_history.pkl"    # Path to save training history

    # Set up callbacks for training:
    # - EarlyStopping: Stop training if validation accuracy doesn't improve for 5 epochs
    # - ModelCheckpoint: Save the best model based on validation accuracy
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',         # Watch validation accuracy
            patience=5,                     # Stop after 5 epochs without improvement
            restore_best_weights=True,      # Restore weights from the best epoch
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_save_path),  # Save model to this path
            monitor='val_accuracy',         # Watch validation accuracy
            save_best_only=True,            # Only save when accuracy improves
            mode='max',                     # Maximize accuracy
            verbose=1
        )
    ]

    # Train the model using the generators
    print("🚀 Starting training...")
    history = model.fit(
        train_gen,                         # Training data generator
        epochs=20,                         # Number of epochs to train
        validation_data=val_gen,           # Validation data generator
        callbacks=callbacks,               # Use callbacks for early stopping and saving
        verbose=1                          # Print training progress
    )

    # Save the training history (accuracy, loss, etc.) to a file using pickle
    import pickle
    with open(str(history_save_path), 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Training history saved to {history_save_path}")

    # Print completion message
    print("✅ Model training completed and saved.")
    return model, history

# If this script is run directly (not imported as a module), execute the following block
if __name__ == "__main__":
    model, history = train_model()