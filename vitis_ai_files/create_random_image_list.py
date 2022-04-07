import os
import glob
import random


if __name__ == "__main__":
    NUM_CALIB = 10_000
    OUTPUT = "ai_calib.txt"
    PATH2TRAINIMGS = "./train_images"
    train_imgs = glob.glob(os.path.join(PATH2TRAINIMGS, "*.jpg"))
    calib_imgs = random.choices(train_imgs, k=NUM_CALIB)

    with open(OUTPUT, "w") as f:
        for c in calib_imgs:
            f.write(c + "\n")
