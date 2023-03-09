import os
import cv2
import glob
import json
import shutil
import argparse

from pathlib import PurePath


def handle_args():
    parser = argparse.ArgumentParser(
        prog="main.py", description="Convert posetrack to yolo format"
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        required=True,
        help="Path to annotation directory",
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output directory"
    )
    return parser.parse_args()


def convert_posetrack_to_yolo(bbox, shape):
    """
    Convert a bounding box from posetrack format to yolo format
    input: bbox = [x1, y1, w, h]
    output: bbox = [xc, yc, w, h] where xc, yc are the center in relative coordinates
    :param bbox: list of 4 floats
    :param shape: list of 2 ints (width, height) from image
    """
    xc = (bbox[0] + bbox[2] / 2) / shape[0]
    yc = (bbox[1] + bbox[3] / 2) / shape[1]
    w = bbox[2] / shape[0]
    h = bbox[3] / shape[1]
    return [xc, yc, w, h]


def get_image_shape(image_path):
    """
    Get the shape of an image
    :param image_path: path to image
    :return: list of 2 ints (width, height)
    """
    image = cv2.imread(image_path)
    return image.shape[1], image.shape[0]


def generate_split(output_dir, data_path, annotation_files):
    """
    Generate a split file for the dataset
    :param output_dir: path to output directory
    :param annotation_files: list of paths to annotations
    """
    os.makedirs(output_dir, exist_ok=True)
    for annotation_file in annotation_files:
        with open(annotation_file, "r") as f:
            ann_dict = json.load(f)
        images = ann_dict["images"]
        image_path_by_id = {
            image["id"]: PurePath(os.path.join(data_path, image["file_name"]))
            for image in images
        }
        image_shape = get_image_shape(os.path.join(data_path, images[0]["file_name"]))
        if "annotations" not in ann_dict:
            continue
        anns = ann_dict["annotations"]
        bboxes_by_id = {image["id"]: [] for image in images}
        for ann in anns:
            yolo_bbox = convert_posetrack_to_yolo(ann["bbox"], image_shape)
            bboxes_by_id[ann["image_id"]].append(yolo_bbox)
        for image_id, bboxes in bboxes_by_id.items():
            if len(bboxes) > 0:
                path = image_path_by_id[image_id]
                file_name = path.parent.name + "_" + path.stem
                shutil.copyfile(path, os.path.join(output_dir, file_name + path.suffix))
                with open(os.path.join(output_dir, file_name + ".txt"), "w") as f:
                    for bbox in bboxes:
                        f.write("1 " + " ".join([str(x) for x in bbox]) + "\n")


def main():
    args = handle_args()

    anns_path = args.annotation_path
    train_anns_path = os.path.join(anns_path, "train")
    val_anns_path = os.path.join(anns_path, "val")
    test_anns_path = os.path.join(anns_path, "test")
    data_path = args.data_path

    train_anns = glob.glob(os.path.join(train_anns_path, "*.json"))
    val_anns = glob.glob(os.path.join(val_anns_path, "*.json"))
    test_anns = glob.glob(os.path.join(test_anns_path, "*.json"))

    output_dir = os.path.join(args.output_path, "train")
    generate_split(output_dir, data_path, train_anns)

    output_dir = os.path.join(args.output_path, "val")
    generate_split(output_dir, data_path, val_anns)

    output_dir = os.path.join(args.output_path, "test")
    generate_split(output_dir, data_path, test_anns)


if __name__ == "__main__":
    main()
