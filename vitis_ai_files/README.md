# Vitis AI Files

After setting with Vitis AI docker environment, access Vitis AI environment using the command:

```bash
./docker_run.sh xilinx/vitis-ai-gpu:latest
# or ./docker_run.sh xilinx/vitis-ai-gpu:other_tagname?
```

Use `conda activate vitis-ai-tensorflow` to access the Vitis AI Tensorflow tools.

**Note** that from now you may need to change `path/to` in the following scripts to make them workable with your environment.

## Convert darknet to .pb

Move to the location where you can `git clone` and `git clone` the `keras-YOLOv3-model-set` repository:

```bash
git clone https://github.com/david8862/keras-YOLOv3-model-set
```

`keras-YOLOv3-model-set` is used in `convert_pb.sh` to convert darknet `.weights` and `.cfg` to `.pb`. Use `convert_pb.sh` to able to process further with the Vitis AI Tensorflow tools.

```bash
bash convert_pb.sh
```

To quantize the weights of `.pb` file requires the calibration dataset. You can use `create_random_image_list.py` to output the `ai_calib.txt`, which contains the list of images for the calibration datasets. Then calibrate and quantize the model with `quant.sh`.

```bash
bash quant.sh
```

After that, compile the quantized model to `.xmodel` with `compile.sh`.

```bash
bash compile.sh
```
