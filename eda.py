# Import the os module for file and directory operations
import os
# Import matplotlib for plotting graphs and images
import matplotlib.pyplot as plt
# Import seaborn for advanced data visualization (bar plots, etc.)
import seaborn as sns
# Import PIL's Image class for opening and manipulating image files
from PIL import Image
# Import pandas for data manipulation and analysis (DataFrames)
import pandas as pd
# Import numpy for numerical operations (not used directly here, but often useful)
import numpy as np

# 🔧 Fix: Dynamically find the project root
SCRIPT_DIR = os.path.dirname(__file__)          # Get the directory of the current script (src/)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)      # Get the parent directory (project root: "Fifth project/")

# Define the path to the assets folder (for saving plots and images)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
# Define the path to the data folder (where image data is stored)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Create the assets folder if it does not already exist
os.makedirs(ASSETS_DIR, exist_ok=True)

def perform_eda(data_dir):
    """Perform Exploratory Data Analysis."""
    # List all subdirectories in the data directory (each represents a class)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # Dictionary to store the number of images in each class
    class_counts = {}

    # Set up a figure for displaying sample images (2 rows x 3 columns)
    plt.figure(figsize=(12, 8))

    # Loop through each class and process its images
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        # List all image files in the class folder with valid extensions
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        # If no images are found in the class folder, print a message and skip
        if not images:
            print(f"No images found in {class_path}")
            continue

        # Store the count of images for this class
        class_counts[class_name] = len(images)

        # Load one sample image from the class folder
        sample_img_path = os.path.join(class_path, images[0])
        img = Image.open(sample_img_path)
        img = img.resize((128, 128))  # Resize the image for display
        # Add the image to the subplot
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img)
        plt.title(f"{class_name} ({len(images)} samples)")
        plt.axis("off")  # Hide axis for image display

    # Adjust layout to prevent overlap and save the figure to the assets folder
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "eda_sample_images.png"))
    plt.show()  # Display the sample images

    # Create a DataFrame to show class distribution (number of images per class)
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    print("\nClass Distribution:")
    print(df)  # Print the class distribution table

    # Plot a bar chart of the class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Class", y="Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)  # Rotate class names for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "class_distribution.png"))  # Save the plot
    plt.show()  # Display the bar chart

    # Return the DataFrame containing class distribution
    return df

# If this script is run directly (not imported as a module), execute the following block
if __name__ == "__main__":
    # Ensure DATA_DIR points to the correct data folder
    df = perform_eda(DATA_DIR)