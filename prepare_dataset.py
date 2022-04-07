import glob
import json
import os
from typing import Any, Dict, Tuple, List

import cv2
from natsort import natsorted
from sklearn.model_selection import train_test_split

# The data without interested classes: (Pedestrain, Car).
BLACKLISTS = [
    "train_16_48",
    "train_23_341",
    "train_23_342",
    "train_23_343",
    "train_23_344",
    "train_23_345",
    "train_23_346",
    "train_23_347",
    "train_23_348",
    "train_23_349",
    "train_23_350",
    "train_23_351",
    "train_23_352",
    "train_23_355",
    "train_23_490",
    "train_23_496",
    "train_11_567",
    "train_11_568",
]


def load_json(json_path: str) -> Dict[str, Any]:
    json_path = os.path.abspath(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_txt(path: str) -> List[Tuple[int, float, float, float]]:
    """Given a txt file and extract a list of (center_x, center_y, width, and height)."""
    assert isinstance(path, str)
    objects = []
    f = open(path, "r")
    for line in f:
        label, cx, cy, w, h = line.strip().split(" ")
        label, cx, cy, w, h = int(label), float(cx), float(cy), float(w), float(h)
        obj = (label, cx, cy, w, h)
        objects.append(obj)
    f.close()
    return objects


def video2images(
    video_path: str,
    save_path: str,
) -> None:
    """Convert a video to images frame by frame. Save these images to `save_path` location.
    Example:
    >>> video2images("test_73.mp4", "test_images")
    """
    assert isinstance(video_path, str)
    assert isinstance(save_path, str)
    assert os.path.isfile(video_path)

    video_path = os.path.abspath(video_path)
    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened is False:
        raise ValueError(f"Cannot open a video {sample_path}.")

    num_frames = 0
    while cap.isOpened():
        _, frame = cap.read()
        if frame is not None:
            fname = video_path.split("/")[-1].split(".")[0]
            save_img = os.path.join(save_path, f"{fname}_{num_frames}.jpg")
            cv2.imwrite(save_img, frame)
        else:
            break
        num_frames += 1
    cap.release()


def generate_data(
    dirpath: str, savepath: str, blacklists: List[str], split: float = 0.20
) -> None:
    """Generate a train.txt and test.txt at `savepath`. This function designed for using
    with AlexeyAB darknet."""
    dirpath = os.path.abspath(dirpath)
    savepath = os.path.abspath(savepath)
    assert os.path.isdir(dirpath)
    assert 0 < split <= 1.0

    datapaths = glob.glob(os.path.join(dirpath, "*.jpg"))
    datatype = datapaths[0].split(".")[-1]
    blacklists = [os.path.join(dirpath, b) + f".{datatype}" for b in blacklists]

    datapaths = list(filter(lambda x: x not in blacklists, datapaths))
    datapaths = natsorted(datapaths)
    train_data, test_data = train_test_split(
        datapaths, test_size=split, random_state=0, shuffle=False
    )
    with open(os.path.join(savepath, "train.txt"), "w") as f:
        for t in train_data:
            f.write(f"{t}\n")

    with open(os.path.join(savepath, "test.txt"), "w") as f:
        for t in test_data:
            f.write(f"{t}\n")


def cvt_xyminmax2cxcywh(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    origin_size: Tuple[int, int] = (1_936, 1_216),
) -> Tuple[float, float, float, float]:
    """Convert from (x_min, y_min, x_max, y_max) to
    (x_center, y_center, height, width) with normalization.
    `image_size` should arrange with (height, width).
    Example:
    >>> cvt_xyminmax2cxcywh(100, 100, 200, 200, (300, 300))
    (0.5, 0.5, 0.3333333333333333, 0.3333333333333333)
    """
    # TODO: support origin_size with None to not normalization.
    assert len(origin_size) == 2, "Expect two values in origin_size for (height, width)"
    width, height = origin_size
    w = abs(xmax - xmin) / width
    h = abs(ymax - ymin) / height
    cx = (w / 2) + (xmin / width)
    cy = (h / 2) + (ymin / height)
    return (cx, cy, w, h)


def cvt_cxcywh2xyminwh(
    cx: int, cy: int, w: int, h: int, origin_size: Tuple[int, int] = (1_936, 1_216)
) -> Tuple[float, float, float, float]:
    """With auto renormalization.
    >>> cvt_cxcywh2xyminwh(0.5, 0.5, 0.3333333333333333, 0.3333333333333333, (300, 300))
    (100, 100, 100, 100)
    """
    # TODO: support origin_size with None to not normalization.
    assert len(origin_size) == 2, "Expect two values in origin_size for (height, width)"
    width, height = origin_size
    # w, h = round(w * width), round(h * height)
    # w, h = w * width, h * height
    xmin = round((cx - (w / 2)) * width)
    ymin = round((cy - (h / 2)) * height)
    return (xmin, ymin, round(w * width), round(h * height))


def jsonlabel2txt(
    json_path: str, txt_path: str, origin_size: Tuple[int, int] = (1_936, 1_216)
) -> None:
    """Convert json label to txt files follows yolo format.
    Assign label 0 to `Pedestrain` and label 1 to `Car`.
    Return a list of name of frames that do not contain labels: `Car` and `Pedestrain`.
    """
    assert len(origin_size) == 2, "Expect two values in origin_size for (height, width)"
    txt_path = os.path.abspath(txt_path)
    os.makedirs(txt_path, exist_ok=True)

    json_file = load_json(json_path)
    fname = json_path.split("/")[-1].split(".")[0]
    # `sequence` contains frame information?
    # `attr` contains two attributes from route and timeofday.
    seq = json_file["sequence"]

    # Signal, Signs, Pedestrain, Car, Motorbike, Bicycle
    # Care only Car and Pedestrain in 5th AI Edge.
    # train seq 600 frames and each frame consists of multiple objects.
    nolabels = []
    for s in range(len(seq)):
        objects = seq[s]
        if "Pedestrian" in objects.keys():
            pedestrians = objects["Pedestrian"]
            LABEL = 0
            for p in range(len(pedestrians)):
                _, box2d = pedestrians[p]["id"], pedestrians[p]["box2d"]
                xmin, ymin, xmax, ymax = box2d
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, origin_size[0] - 1)
                ymax = min(ymax, origin_size[1] - 1)

                cx, cy, w, h = cvt_xyminmax2cxcywh(
                    xmin, ymin, xmax, ymax, (origin_size[0], origin_size[1])
                )
                nametxt = fname + "_" + str(s) + ".txt"
                nametxt = os.path.join(txt_path, nametxt)
                with open(nametxt, "a") as f:
                    f.write(f"{LABEL} {cx} {cy} {w} {h}\n")

        if "Car" in objects.keys():
            cars = objects["Car"]
            LABEL = 1
            for c in range(len(cars)):
                _, box2d = cars[c]["id"], cars[c]["box2d"]
                # box2d in format of x0, y0, x1, y1 format.
                # Ex: [636, 563, 660, 626]
                # id_ for object tracking? There are some duplication in all list.
                xmin, ymin, xmax, ymax = box2d
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, origin_size[0] - 1)
                ymax = min(ymax, origin_size[1] - 1)

                cx, cy, w, h = cvt_xyminmax2cxcywh(
                    xmin, ymin, xmax, ymax, (origin_size[0], origin_size[1])
                )
                nametxt = fname + "_" + str(s) + ".txt"
                nametxt = os.path.join(txt_path, nametxt)
                with open(nametxt, "a") as f:
                    f.write(f"{LABEL} {cx} {cy} {w} {h}\n")

        if not "Pedestrian" in objects.keys() and not "Car" in objects.keys():
            name_nolabel = fname + "_" + str(s)
            nolabels.append(name_nolabel)
    print(
        "Detect number of frames without Pedestrian or Car labels in"
        f"{os.path.basename(json_path)}: {len(nolabels)}."
    )
    return nolabels


if __name__ == "__main__":
    annotation_path = "./dataset/train_annotations"
    label_path = "./dataset/train_images"
    train_video_path = "./dataset/train_videos"
    train_image_path = "./dataset/train_images"
    test_video_path = "./dataset/test_videos"
    test_image_path = "./dataset/test_images"

    annotation_path = os.path.expanduser(annotation_path)
    json_paths = glob.glob(os.path.join(annotation_path, "*.json"))

    print("Converting .json labels to .txt.")
    blacklists = []
    for j in json_paths:
        blacklists += jsonlabel2txt(j, label_path)
    blacklists = [b + ".jpg" for b in blacklists]

    print("Converting train videos to train images.")
    train_video_path = os.path.expanduser(train_video_path)
    train_video_paths = glob.glob(os.path.join(train_video_path, "*.mp4"))
    for v in train_video_paths:
        video2images(v, train_image_path)

    print("Converting test videos to test images.")
    test_video_path = os.path.expanduser(test_video_path)
    test_video_paths = glob.glob(os.path.join(test_video_path, "*.mp4"))
    for v in test_video_paths:
        video2images(v, test_image_path)
    print("Generate labels for DarkNet")
    generate_data("./dataset/train_images", "./dataset/", BLACKLISTS)
