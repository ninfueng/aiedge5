#!/usr/bin/env python
# coding: utf-8
import os
import glob

import cv2
import numpy as np
from natsort import natsorted
from pynq_dpu import DpuOverlay

from utils import post_process, non_max_suppression_np

if __name__ == "__main__":
    conf_thres = 0.8
    nms_thres = 0.4
    num_anchors = 3

    np.random.seed(0)
    num2classes = {0: "Pedestrian", 1: "Car"}
    num_classes = len(num2classes)

    overlay = DpuOverlay("./dpu/dpu.bit")
    overlay.load_model("./xmodel/yolov4_tiny_512.xmodel")
    dpu = overlay.runner
    in_tensors = dpu.get_input_tensors()
    out_tensors = dpu.get_output_tensors()

    out_placeholders = []
    for in_tensor in in_tensors:
        in_placeholder = np.empty(in_tensor.dims, dtype=np.float32)
        print(f"Input dims: {in_tensor.dims}")
    assert in_tensor.dims[1] == in_tensor.dims[2]
    img_size = in_tensor.dims[1]

    for out_tensor in out_tensors:
        out_placeholders.append(np.empty(out_tensor.dims, dtype=np.float32))
        print(f"Output dims: {out_tensor.dims}")

    anchors = [(5, 14), (9, 30), (27, 29), (16, 64), (49, 68), (100, 117)]
    mask_anchors0 = [3, 4, 5]
    mask_anchors1 = [1, 2, 3]
    anchors0 = [anchors[i] for i in mask_anchors0]
    anchors1 = [anchors[i] for i in mask_anchors1]
    num_anchors = len(mask_anchors0)
    assert len(mask_anchors0) == len(mask_anchors1)

    test_videos = glob.glob("./dataset/test_videos/*.mp4")
    test_videos = natsorted(test_videos)

    submit_json = {}
    frame_counter = 0
    detection_list = []
    for test_video in test_videos:
        print(f"Processing on {test_video}")
        video_name = os.path.basename(test_video)
        submit_json.update({video_name: []})
        video = cv2.VideoCapture(test_video)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if video.isOpened():
            video_width, video_height = video.get(3), video.get(4)
        while True:
            ret, o_frame = video.read()
            frame_counter += 1
            if not ret:
                break

            o_frame = cv2.cvtColor(o_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                o_frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR
            )
            input_img = frame.astype(np.float32) / 255.0

            input_img = np.expand_dims(input_img, 0)
            job_id = dpu.execute_async(input_img, out_placeholders)
            dpu.wait(job_id)

            det0, det1 = out_placeholders
            det0 = post_process(det0, anchors0)
            det1 = post_process(det1, anchors1)

            det = np.concatenate((det0, det1), axis=1)
            detections = non_max_suppression_np(
                det, num_classes, conf_thres, nms_thres
            )[0]
            detection_list.append(detections)
    np.save("detection_list.npy", detection_list, allow_pickle=True)

    del overlay
    del dpu
