## Dataset download from https://github.com/PlumSmile/Dog-Emotion-Dataset

import os
import sys
import shutil
import random
import zipfile
import argparse
import logging
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Constants
github_zip_url = (
    "https://github.com/PlumSmile/Dog-Emotion-Dataset/archive/refs/heads/main.zip"
)

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("get_PlumSmile_data.log")


def download_dataset(zip_url, download_path):
    if download_path.exists():
        logger.info(f"Archive already present at {download_path}")
        return

    logger.info(f"Downloading dataset from {zip_url} ...")

    req = Request(zip_url, headers={"User-Agent": "python-urllib"})

    try:
        with urlopen(req) as resp, open(download_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        logger.info("Download complete.")

    except (URLError, HTTPError) as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)


def extract_zip(zip_path, extract_to):
    logger.info(f"Extracting {zip_path} -> {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    logger.info("Extraction complete.")


def prepare_splits(src_train, src_test, dst_root):
    for split in ("train", "val", "test"):
        logger.info(f"Preparing {split} set...")
        classes = os.listdir(src_train if split != "test" else src_test)

        for cls in sorted(classes):
            src_dir = (src_train if split != "test" else src_test) / cls
            dst_dir = dst_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            if split in ("train", "val"):
                all_imgs = sorted(src_dir.glob("*"))
                n_val = int(len(all_imgs) * 0.25)
                val_imgs = set(random.sample(all_imgs, n_val))
                imgs_to_copy = (
                    val_imgs
                    if split == "val"
                    else [p for p in all_imgs if p not in val_imgs]
                )
            else:
                imgs_to_copy = sorted(src_dir.glob("*"))

            for img_path in imgs_to_copy:
                shutil.copy(img_path, dst_dir / img_path.name)

        logger.info(f"{split.capitalize()} set done, total classes: {len(classes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Dog Emotion dataset for PyTorch DataLoader"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./PlumSmile_data/dog_emotions"),
        help="Root directory where train/val/test folders will be created",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work"),
        help="Temporary working directory for downloads and extraction",
    )

    parser.add_argument(
        "-d", "--debug", dest="debug", help="Enable debug mode", action="store_true"
    )

    args = parser.parse_args()

    logger_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename="get_data.log", level=logger_level)

    console_handler.setLevel(logger_level)
    file_handler.setLevel(logger_level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logger_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prepare dirs
    args.work_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.work_dir / "dog_emotions.zip"
    extract_dir = args.work_dir / "extracted"

    # Download and extract
    download_dataset(github_zip_url, zip_path)
    extract_zip(zip_path, extract_dir)

    # Source paths inside extracted
    base = extract_dir / "Dog-Emotion-Dataset-main" / "Dataset"
    src_train = base / "train"
    src_test = base / "test"

    # Prepare splits
    prepare_splits(src_train, src_test, args.output_dir)

    shutil.rmtree(args.work_dir)

    logger.info(
        f"Done! Data ready at {args.output_dir}/{{train,val,test}}/<class>/*.jpg"
    )


if __name__ == "__main__":
    main()
