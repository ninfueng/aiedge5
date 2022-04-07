python path/to/keras-YOLOv3-model-set/tools/model_converter/convert.py --yolo4_reorder path/to.cfg path/to.weights yolov4-tiny-512.h5

python path/to/keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model yolov4-tiny-512.h5 --output_model=yolov4-tiny-512.pb
