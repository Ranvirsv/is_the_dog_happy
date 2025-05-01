import os
import logging
import argparse
import sys
from ultralytics import YOLO
import torch

## REF: https://docs.ultralytics.com/modes/train/#usage-examples

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("get_annotations.log")


def main(source_dir, target_dir, face_only_mode, oxford_mode):
    """
    Takes the processed kaggle dataset and annotates it using the trained yolo model
    and stores bounding boxes (without classes) in a text file.

    Args:
        source_dir (str): The path to the processed kaggle dataset
        target_dir (str): The path to store the annotations
        face_only_mode (bool): If True, only draws bounding boxes for the face of the dog

    Returns:
        None
    """

    logger.info("Loading Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Model on device: {}".format(device))

    path = "./yolo_model/yolo_model_files"
    model_file = (
        "oxford_model_using_imagenet_weights/best.pt"
        if face_only_mode
        else (
            "oxford_model/best.pt"
            if (face_only_mode and oxford_mode)
            else "imagenet_model/best.pt"
        )
    )
    model_path = os.path.join(path, model_file)

    logger.info("Using {}".format(model_file))

    model = YOLO(model_path)

    model.to(device)
    model.eval()

    logger.info("Creating Target Sub-Dir...")

    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, subset), exist_ok=True)

    logger.info("Loading Source Directory...")

    for subset in ["train", "val", "test"]:
        subset_path = os.path.join(source_dir, subset)
        if not os.path.isdir(subset_path):
            continue

        # replicate the category folders(Angry, Happy, Relaxed, Sad) in target_dir
        for category_name in os.listdir(subset_path):
            category_path = os.path.join(subset_path, category_name)
            if not os.path.isdir(category_path):
                continue

            # annotations/train/angry
            target_category_path = os.path.join(target_dir, subset, category_name)
            os.makedirs(target_category_path, exist_ok=True)

            logger.info("Generating Annotations...")

            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                results = model(image_path)
                r = results[0]

                # Now, create a text file to store bounding boxes only
                base_name, _ = os.path.splitext(image_name)
                bbox_txt_path = os.path.join(target_category_path, f"{base_name}.txt")

                with open(bbox_txt_path, "w") as f:
                    for box in r.boxes:
                        # box.xyxy -> shape [1,4], e.g. [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        f.write(f"{x1} {y1} {x2} {y2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="get_aotated_data.py",
        description="Annotate the kaggle dataset",
        epilog="This script annotates the dataset from kaggle",
    )

    parser.add_argument(
        "-f",
        "--face_only",
        dest="face_only",
        help="Enable face_only mode, only draws bounding boxes for the face of the dog",
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--oxford_only",
        dest="oxford_only",
        help="Enable oxford_only mode, only draws bounding boxes for the face of the dog using model only trained on oxford data",
        action="store_true",
    )

    parser.add_argument(
        "-d", "--debug", dest="debug", help="Enable debug mode", action="store_true"
    )

    args = parser.parse_args()

    logger_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename="get_annotations.log", level=logger_level)

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

    source_dir = os.path.join(os.getcwd(), "data")
    target_dir = (
        "./annotations_face_oxford_only/"
        if (args.face_only and args.oxford_only)
        else "./annotations_face_combined/" if args.face_only else "./annotations/"
    )
    os.makedirs(target_dir, exist_ok=True)

    logger.info("Starting Anotation Script")
    logger.info("Face only mode: {}".format(args.face_only))

    main(source_dir, target_dir, args.face_only, args.oxford_only)

    logger.info(f"Annotated dataset stored in {target_dir}")
