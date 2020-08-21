#!/usr/bin/env python3

import struct
import numpy as np
import os
import sys
from PIL import Image

assert(len(sys.argv) == 2)
filepath = sys.argv[1]

RECORD_SIZE = 2052
i = 0
print("Reading {}".format(filepath))
with open(filepath, 'rb') as f:
    filename = os.path.basename(filepath)
    while True:
        s = f.read(RECORD_SIZE)
        if s is None or len(s) < RECORD_SIZE:
            break
        r = struct.unpack(">H2sHBBBBBBIHHHHBBBBHH2016s4x", s)
        img = Image.frombytes('F', (64, 63), r[20], 'bit', (4, 0))
        img = img.convert('L')
        img = img.point(lambda x: 255 - (x << 4))
        i = i + 1
        dirname = r[1].decode('utf-8')
        dirname = dirname.replace('\0', '')
        dirname = dirname.replace(' ', '')
        dirname = dirname.replace('\\', 'YEN')
        dirname = dirname.replace('+', 'PLUS')
        dirname = dirname.replace('-', 'MINUS')
        dirname = dirname.replace('*', 'ASTERISK')
        try:
            os.makedirs(f"data/extract/{dirname}")
        except:
            pass
        imagefile = f"data/extract/{dirname}/{filename}_{i:0>6}.png"
        print(imagefile)
        img.save(imagefile)
    