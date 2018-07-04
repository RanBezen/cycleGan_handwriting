from PIL import Image, ImageDraw, ImageFont
import numpy as np

def str_to_image(str):
    im = []
    img = Image.new('L', (512, 48),255)
    fnt = ImageFont.truetype('fonts/ARIALN.TTF', 30)
    d = ImageDraw.Draw(img)
    d.text((5, 5), str, font=fnt, fill=(0))
    imA = np.asarray(img,np.float)
    im.append(imA)
    im = np.array(im) / 127.5 - 1.
    im = np.expand_dims(im, axis=3)
    return im
