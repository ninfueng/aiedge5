import numpy as np
import os
import json
import glob

from natsort import natsorted
import numpy as np

from sort.sort import Sort

if __name__ == "__main__":
    IMG_SIZE = 512
    NUM_FRAME_PER_VIDEO = 150
    IMG_HEIGHT, IMG_WIDTH = 1_216, 1_936

    test_videos = glob.glob("./dataset/test_videos/*.mp4")
    test_videos = [os.path.basename(t) for t in test_videos]
    test_videos = natsorted(test_videos)

    submit_json = {}
    mot_tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.5)
    video_idx = 0

    detection_list = np.load("detection_list.npy", allow_pickle=True)
    video_name = test_videos[video_idx]
    print(f"Processing on {video_name}")

    submit_json.update({video_name: []})
    for idx, detections in enumerate(detection_list):
        if idx % NUM_FRAME_PER_VIDEO == 0 and idx != 0:
            mot_tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.5)
            video_idx += 1
            video_name = test_videos[video_idx]
            print(f"Processing on {video_name}")
            submit_json.update({video_name: []})
        json_frame = {"Pedestrian": [], "Car": []}

        if detections is not None:
            # Sort and split based on the detection classes.
            detections = detections[detections[..., -1].argsort()]
            detections0 = detections[detections[:, -1] == 0]
            detections1 = detections[detections[:, -1] == 1]

            tracked_objects = mot_tracker.update(detections0)
            for x1, y1, x2, y2, obj_id in tracked_objects:
                box_h = int(((y2 - y1) / IMG_SIZE) * IMG_HEIGHT)
                box_w = int(((x2 - x1) / IMG_SIZE) * IMG_WIDTH)
                x1 = int((x1 / IMG_SIZE) * IMG_WIDTH)
                y1 = int((y1 / IMG_SIZE) * IMG_HEIGHT)
                x2 = int((x2 / IMG_SIZE) * IMG_WIDTH)
                y2 = int((y2 / IMG_SIZE) * IMG_HEIGHT)
                json_frame["Pedestrian"].append(
                    {"id": int(obj_id), "box2d": [x1, y1, x2, y2]}
                )

            tracked_objects = mot_tracker.update(detections1)
            for x1, y1, x2, y2, obj_id in tracked_objects:
                box_h = int(((y2 - y1) / IMG_SIZE) * IMG_HEIGHT)
                box_w = int(((x2 - x1) / IMG_SIZE) * IMG_WIDTH)
                x1 = int((x1 / IMG_SIZE) * IMG_WIDTH)
                y1 = int((y1 / IMG_SIZE) * IMG_HEIGHT)
                x2 = int((x2 / IMG_SIZE) * IMG_WIDTH)
                y2 = int((y2 / IMG_SIZE) * IMG_HEIGHT)
                json_frame["Car"].append({"id": int(obj_id), "box2d": [x1, y1, x2, y2]})
        submit_json[video_name].append(json_frame)

    with open("submission.json", "w") as f:
        json.dump(submit_json, f)
