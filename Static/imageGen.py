import os
from os import listdir
from PIL import Image
import numpy as np
from Static import HEIGHT as h, WIDTH as w, PATH_EXE

def image_gen(path):
    images = []
    for f in listdir(path):
        with open(os.path.join(path, f), 'rb') as img_set:
            img_arr = img_set.read(h * w)
            while img_arr:
                if len(img_arr) == h * w and img_arr not in images:
                    images.append(img_arr)
                img_arr = img_set.read(h * w)


    count = 0
    for img in images:
        png = Image.fromarray(np.reshape(list(img), (h, w)).astype('float32'), mode='L')
        png.save('image_l%d.png' % count)
        count += 1
    print("#generation done")

if __name__ == "__main__":
    image_gen(PATH_EXE)