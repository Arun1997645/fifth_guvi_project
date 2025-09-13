# Import the os module for interacting with the operating system (file and directory operations)
import os
# Import the Image class from PIL (Python Imaging Library) to work with image files
from PIL import Image
# Import logging to record messages for debugging and tracking the script's progress
import logging

# Setup logging configuration to display INFO level messages and above
logging.basicConfig(level=logging.INFO)
# Create a logger object for this module
logger = logging.getLogger(__name__)

def is_valid_image(filepath):
    """Check if file is a valid image."""
    # Try to open the image file and verify its integrity
    try:
        img = Image.open(filepath)
        img.verify()  # Checks if the image is corrupted
        return True   # If no exception, the image is valid
    except Exception as e:
        # If an error occurs, log the error and return False
        logger.error(f"Invalid image: {filepath}, Error: {e}")
        return False

def clean_data(data_dir):
    """Remove corrupted or invalid images. Skip subdirectories."""
    # Log the start of the data cleaning process
    logger.info("Starting data cleaning...")
    # Define valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # Counter for the number of files removed
    removed_count = 0

    # Loop through each item (class folder) in the data directory
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        # Skip if the item is not a directory (i.e., not a class folder)
        if not os.path.isdir(class_path):
            continue

        # Log which class folder is being cleaned
        logger.info(f"Cleaning class: {class_name}")
        # Loop through each file in the class folder
        for item in os.listdir(class_path):
            item_path = os.path.join(class_path, item)

            # 🔑 Skip if it's a directory (should only process files)
            if os.path.isdir(item_path):
                logger.warning(f"Skipping directory: {item_path}")
                continue

            # Skip and remove files that do not have a valid image extension
            if not item.lower().endswith(valid_extensions):
                logger.warning(f"Removing non-image file: {item_path}")
                try:
                    os.remove(item_path)  # Remove the non-image file
                    removed_count += 1    # Increment the removed files counter
                except Exception as e:
                    # Log any error that occurs while removing the file
                    logger.error(f"Could not remove {item_path}: {e}")
                continue

            # Check if the image file is corrupted or invalid
            if not is_valid_image(item_path):
                logger.warning(f"Removing corrupted image: {item_path}")
                try:
                    os.remove(item_path)  # Remove the corrupted image file
                    removed_count += 1    # Increment the removed files counter
                except Exception as e:
                    # Log any error that occurs while removing the file
                    logger.error(f"Could not remove {item_path}: {e}")

    # Log the completion of data cleaning and the total number of files removed
    logger.info(f"Data cleaning completed. Removed {removed_count} invalid files.")

# If this script is run directly (not imported as a module), execute the following block
if __name__ == "__main__":
    # Define the path to the data directory containing image class folders
    DATA_DIR = r"C:\Users\ARURAVI\PM6\AIML GUVI Projects\Fifth project\data"
    # Call the clean_data function to start cleaning the data directory
    clean_data(DATA_DIR)