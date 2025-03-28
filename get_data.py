from sklearn.model_selection import train_test_split
import shutil
import os
import logging
import argparse
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import sys
print(sys.executable)
print(sys.path)


logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("get_data.log")

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
api = KaggleApi()
api.authenticate()


def main(source_dir, dataset, target_dir, logger):
    logger.info(f"Dataset downloaded to {source_dir}")

    # Download latest version
    api.dataset_download_files(dataset, path=source_dir, unzip=True)

    logger.info("Restructuring dataset...")

    csv_filename = 'dataset_labels.csv'

    data_entries = []

    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, subset), exist_ok=True)

    # Now let's pre-process the data
    # The data is in 4 different folders (sad, angry, happy, relaxed)
    # We will like to process the images so that we have a labeled dataset

    train_size = 0.6    # 60% for training
    # 50% of remaing 40% data for validation, 50% for testing (20% of total data in each set)
    val_ratio = 0.5

    image_dir = os.path.join(source_dir, 'images')

    for emotion in os.listdir(image_dir):
        emotion_dir = os.path.join(image_dir, emotion)
        if not os.path.isdir(emotion_dir):
            logger.info(f"Skipping non-directory file: {emotion_dir}")
            continue

        # List all image files (adjust extensions if needed)
        image_files = [
            os.path.join(emotion_dir, f)
            for f in os.listdir(emotion_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            logger.warning(f"No image files found in {emotion_dir}")
            continue

        # First, split into training and temporary (for validation+test)
        train_files, temp_files = train_test_split(image_files,
                                                   train_size=train_size,
                                                   random_state=42)

        # Then, split temporary into validation and test sets
        val_files, test_files = train_test_split(temp_files,
                                                 train_size=val_ratio,
                                                 random_state=42)

        # Define target paths for this emotion
        for subset, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            target_emotion_dir = os.path.join(target_dir, subset, emotion)
            os.makedirs(target_emotion_dir, exist_ok=True)

            for file_path in files:
                file_name = os.path.basename(file_path)
                destination = os.path.join(target_emotion_dir, file_name)

                shutil.copy(file_path, destination)

                data_entries.append({
                    'file_path': destination,
                    'label': emotion,
                    'split': subset
                })

        logger.info(
            f"{emotion}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images")

    logger.info("Restructuring complete!")

    df = pd.DataFrame(data_entries)
    csv_path = os.path.join(target_dir, csv_filename)

    df.to_csv(csv_path, index=False)
    logger.info(f"Dataset saved to {csv_path}")

    logger.info(f"Deleating {image_dir}")
    shutil.rmtree(source_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="get_data.py",
        description="Download dataset from kaggle",
        epilog="This script downloads the dataset from kaggle",
    )

    parser.add_argument("-d", "--debug", dest="debug",
                        help="Enable debug mode", action="store_true")

    args = parser.parse_args()

    logger_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename='get_data.log', level=logger_level)

    # Set log level for handlers
    console_handler.setLevel(logger_level)
    file_handler.setLevel(logger_level)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Get the root logger and add handlers
    logger = logging.getLogger()
    logger.setLevel(logger_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    dataset = "devzohaib/dog-emotions-prediction"
    source_dir = os.path.join(os.getcwd(), 'kaggle_data')
    os.makedirs(source_dir, exist_ok=True)

    target_dir = './data/'
    os.makedirs(target_dir, exist_ok=True)

    main(source_dir, dataset, target_dir, logger)

    logger.info(f"Split up data stored in {target_dir}")
