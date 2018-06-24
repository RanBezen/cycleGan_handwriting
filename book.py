from PIL import Image
import glob, os
import numpy as np

def crop():
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
            img=im_resize.crop((0,i,500,i+30))
            a4im = Image.new('L',
                             (512, 48),  # A4 at 72dpi
                             (255))  # White
            a4im.paste(img, img.getbbox())  # Not centered, top-left corner
            # img=a4im.filter(ImageFilter.GaussianBlur(1))
            # threshold = 200
            # img = a4im.point(lambda p: p > threshold and 255)
            img=a4im
            #img=img.resize([800,48])
            dir = 'datasets/sentences/trainA/'+ file + str(indx)
            img.save(dir + ".jpg")

            size = os.path.getsize(dir + ".jpg")
            if size <= 1000:
                os.remove(dir + ".jpg", dir_fd=None)
            else:
                indx=indx+1



"""
img = Image.open('book/book-04.jpg')
cropped_img = img.crop((127, 124, 1180, 2120))
im_resize=cropped_img.resize([1053,2040])
for i in np.arange(0, 2040, 34):
    img = im_resize.crop((0, i, 1053, i + 34))
    dir = 'crop/'
    img.save('book/crop/book-04'+str(i)+'.jpg')
    size=os.path.getsize('book/crop/book-04'+str(i)+'.jpg')
    if size<=1000:
        os.remove('book/crop/book-04'+str(i)+'.jpg', dir_fd=None)
"""
crop()
