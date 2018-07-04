from PIL import Image
import glob, os
import numpy as np

def crop():
    #this function crop the lines from the book pages
    for infile in glob.glob1('book2/', '*.jpg'):
        # print(infile)
        file, ext = os.path.splitext(infile)
        print(file)

        img = Image.open('book2/' + infile)
        # print("pass")
        cropped_img = img.crop((108, 117, 608, 1458))
        im_resize = cropped_img.resize([500, 1320])
        indx=0
        for i in np.arange(0,1320,30):
            #cut the line for all range of 30 pixels
            img=im_resize.crop((0,i,500,i+30))
            #add padding for keep high quality of the sentence
            a4im = Image.new('L',
                             (512, 48),  # A4 at 72dpi
                             (255))  # White
            a4im.paste(img, img.getbbox())  # Not centered, top-left corner
            img=a4im
            dir = 'datasets/sentences/trainA/'+ file + str(indx)
            img.save(dir + ".jpg")

            #if there is a blank line- the images will not save
            size = os.path.getsize(dir + ".jpg")
            if size <= 1000:
                os.remove(dir + ".jpg", dir_fd=None)
            else:
                indx=indx+1


crop()
