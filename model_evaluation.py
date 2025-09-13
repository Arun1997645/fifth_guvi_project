# src/model_evaluation.py
"""
SolarGuard: Model Evaluation Script
Evaluates trained model on validation data and generates performance metrics.
"""

# Import the os module for environment variable and file operations
import os
# Suppress oneDNN warnings (optional, helps reduce TensorFlow log clutter)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting graphs and images
import matplotlib.pyplot as plt
# Import seaborn for advanced data visualization (heatmaps, etc.)
import seaborn as sns
# Import metrics functions from scikit-learn for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
# Import Keras function to load a saved model
from tensorflow.keras.models import load_model
# Import pickle for loading/saving Python objects (not used directly here)
import pickle
# Import Path from pathlib for handling file paths
from pathlib import Path

# Add project root to path so that imports from src/ work correctly
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the data generator creation function from preprocessing script
from src.preprocessing import create_generators

def evaluate_model():
    # Define paths for model and assets
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "best_cnn_model.h5"  # Path to the trained model
    ASSETS_DIR = PROJECT_ROOT / "assets"                        # Path to save evaluation plots
    ASSETS_DIR.mkdir(exist_ok=True)                             # Create assets directory if it doesn't exist

    # Load the trained model from disk
    print(f"Loading model from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Recreate the validation data generator
    DATA_DIR = PROJECT_ROOT / "data"
    print("Loading validation data...")
    _, val_gen = create_generators(DATA_DIR)  # Only need the validation generator

    # Make predictions on the validation set
    print("Predicting on validation set...")
    val_gen.reset()  # Reset generator to start from the beginning
    predictions = model.predict(val_gen, steps=len(val_gen))  # Get predicted probabilities for each class
    y_pred = np.argmax(predictions, axis=1)  # Convert probabilities to predicted class indices
    y_true = val_gen.classes                  # True class labels from the generator

    # Get class labels (names) in the correct order
    class_indices = val_gen.class_indices
    labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    # Print a detailed classification report (precision, recall, f1-score for each class)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = ASSETS_DIR / "confusion_matrix.png"  # Path to save confusion matrix plot
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion Matrix saved to {cm_path}")

    # Calculate overall metrics: accuracy, precision, recall, f1-score (weighted average)
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print the calculated metrics
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# If this script is run directly (not imported as a module), execute the following block
if __name__ == "__main__":
    evaluate_model()