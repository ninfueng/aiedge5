# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np

calib_image_list = "./ai_calib.txt"
calib_batch_size = 10
size = 512, 512

# normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0  # could be also 256.0


def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()

    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        filename = calib_image_name
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        images.append(image)
        print(
            "Iteration number : {} and index number {} and  file name  {} ".format(
                iter, index, filename
            )
        )
    return {"image_input": images}


#######################################################
def main():
    calib_input(0)
    calib_input(1)


if __name__ == "__main__":
    main()
