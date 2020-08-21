from PIL import Image
import sys
import os
import json
import glob
import random

def get_concat_h_cut(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_cut(im1, im2):
    dst = Image.new('L', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


order_N = int(sys.argv[1])
file_N = int(sys.argv[2])
output_json = sys.argv[3]
outputs = []
try:
    os.makedirs("data/extract/number")
except:
    pass
for i in range(file_N):
    number = random.randint(0, 10 ** order_N)
    number = str(number).zfill(order_N)
    dst = None
    for j, n in enumerate(list(number)):
        img_files = glob.glob(f"data/extract/{n}/*.png")
        img_file = random.choice(img_files)
        img = Image.open(img_file)
        if j == 0:
            dst = img
        else:
            dst = get_concat_h_cut(dst, img)

    imagefile = f"data/extract/number/{number}_{i:0>6}.png"
    print(imagefile)
    outputs.append((imagefile, str(number)))
    dst.save(imagefile)

with open(output_json, 'w') as fo:
    json.dump(outputs, fo, ensure_ascii=False, indent=4)
