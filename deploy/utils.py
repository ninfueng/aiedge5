import signal

import numpy as np
from pynq import Overlay
from typing import List

from isa import lui, load_instructs

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def timeout_handler(signum, frame) -> None:
    raise TimeoutError(f"Timeout, SIGNUM: {signum}, frame: {frame}")


signal.signal(signal.SIGALRM, timeout_handler)


def run_riscv(
    vars_: List[int],
    bitdir: str = "./bitstream/riscv.bit",
    instruct_dir: str = "main.hex",
    timeout: int = 5,
):
    """Instantate and run riscv core given ISAs in the hex format."""
    signal.alarm(timeout)
    # Initialize stack pointer to some place. Avoid stack overflow problems.
    start_sp = lui(2, 0xA0030000)
    instructs = load_instructs(instruct_dir)
    instructs.insert(0, start_sp)

    try:
        overlay = Overlay(bitdir)
        imem = overlay.IMEM_CONTROL
        dmem = overlay.DMEM_CONTROL
        gpio = overlay.axi_gpio_0

        for idx, i in enumerate(instructs):
            imem.write(idx * 4, i)
        for idx, v in enumerate(vars_):
            dmem.write(idx * 4, v)
        gpio.write(0x00, 1)
        gpio.write(0x00, 0)
        hw_flag = True

    except TimeoutError:
        print(
            f"TIMEOUT! Run RISCV assembly longer than {timeout} seconds,"
            "using the ARM instead."
        )
        hw_flag = False

    signal.alarm(0)
    results = ()
    if hw_flag:
        x = (dmem.read(4 * 7), dmem.read(4 * 9))
        y = (dmem.read(4 * 8), dmem.read(4 * 10))
        results = (x, y)
    return hw_flag, results


def bbox_iou_np(box1, box2):
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (
        np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1)
        + area
        - iw * ih
    )
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih
    return intersection / ua


def post_process(yolo_out, anchors, num_classes=2, img_size=512):
    num_anchors = len(anchors)
    b, wh, _, _ = yolo_out.shape
    yolo_out = yolo_out.reshape(yolo_out.shape[:-1] + (num_anchors, num_classes + 5))
    cx = sigmoid(yolo_out[..., 0])
    cy = sigmoid(yolo_out[..., 1])
    w = yolo_out[..., 2]
    h = yolo_out[..., 3]
    conf = sigmoid(yolo_out[..., 4])
    cls = sigmoid(yolo_out[..., 5:])

    # TODO: why switching this x_offset and y_offset makes it works!!!!!
    x_offset = np.concatenate([np.arange(wh) for _ in range(wh)], axis=0).reshape(
        1, wh, wh, 1
    )
    y_offset = x_offset.T

    scale = img_size / wh
    scaled_anchors = np.array([(aw / scale, ah / scale) for aw, ah in anchors])
    anchor_w = scaled_anchors[:, 0].reshape((1, 1, 1, num_anchors))
    anchor_h = scaled_anchors[:, 1].reshape((1, 1, 1, num_anchors))

    bbs = np.zeros_like(yolo_out[..., :4], dtype=np.float32)
    bbs[..., 0] = cx + x_offset
    bbs[..., 1] = cy + y_offset
    bbs[..., 2] = np.exp(w) * anchor_w
    bbs[..., 3] = np.exp(h) * anchor_h

    output = np.concatenate(
        (
            bbs.reshape(b, -1, 4) * scale,
            conf.reshape(b, -1, 1),
            cls.reshape(b, -1, num_classes),
        ),
        -1,
    )
    return output


def non_max_suppression_np(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    x1 = prediction[:, :, 0] - prediction[:, :, 2] / 2
    y1 = prediction[:, :, 1] - prediction[:, :, 3] / 2
    x2 = prediction[:, :, 0] + prediction[:, :, 2] / 2
    y2 = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, 0] = x1
    prediction[:, :, 1] = y1
    prediction[:, :, 2] = x2
    prediction[:, :, 3] = y2

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.shape[0]:
            continue

        class_conf = np.max(image_pred[:, 5 : 5 + num_classes], 1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5 : 5 + num_classes], 1)
        class_pred = np.expand_dims(class_pred, -1)
        detections = np.concatenate(
            (
                image_pred[:, :5],
                class_conf.astype(np.float32),
                class_pred.astype(np.float32),
            ),
            1,
        )
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            conf_sort_index = np.argsort(detections_class[:, 4])[::-1]
            detections_class = detections_class[conf_sort_index]

            max_detections = []
            while detections_class.shape[0]:
                tmp = np.expand_dims(detections_class[0], 0)
                max_detections.append(tmp)
                if len(detections_class) == 1:
                    break
                ious = bbox_iou_np(max_detections[-1], detections_class[1:])[0]
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = np.concatenate(max_detections)
            output[image_i] = (
                max_detections
                if output[image_i] is None
                else np.concatenate((output[image_i], max_detections))
            )
    return output
