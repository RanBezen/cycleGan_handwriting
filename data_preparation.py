import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
from PIL import Image
import glob, os


def prepare_data(img_dir):
    #create list of the images dir name
    image_dirs = np.array([dirpath for (dirpath, dirnames, filenames) in gfile.Walk(os.getcwd()+'/'+img_dir)])
    file_list = []
    extensions = ['jpg', 'JPEG']
    dir_name = os.path.basename(img_dir)
    image_file_list =[]
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
                # Building the filename pattern
                file_glob = os.path.join(img_dir,'*.' + extension)
                #This looks for a file name pattern, helps us ensure that only jpg extensions are choosen
                image_file_list = gfile.Glob(file_glob)
                file_list.extend(image_file_list)
    return file_list


def read_image_array_np_GRAYSCALE(image_loc_array,w,h):
    #create a numpy array of images from the paths array
    resized_image_array=[]
    i=0
    for image_loc in image_loc_array:
        image_decoded = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)
        image_resize=cv2.resize(image_decoded,(w,h))
        print(image_resize.shape)
        resized_image_array.append(image_resize)
        print(image_loc)
        print('sample number:',i)
        i=i+1
    resized_image_array=np.asarray(resized_image_array)
    return resized_image_array

def create_npy_file(arr,name):

    np.save("data/"+name, arr)
    print("npy dataset created", arr.shape, "   ", type(arr))



def create_large_folder(dir,new_dir):
    for x in os.walk(dir):
        if x[2]:
            #changeDir(x, new_dir)
            changeDirAndProcess(x, new_dir)
            print(x[2])


def changeDirAndProcess(x,newPath):
    for i in x[2]:
        dir=x[0]+'/'+i
        file, ext = os.path.splitext(i)
        #print("old",dir)
        newDir=newPath+file+'.jpg'
        image_preperation(dir,newDir)


def image_preperation(im_pth, newPath):

    im = Image.open(im_pth)
    w, h = im.size
    if w > 1500 and h <= 150:
        im = im.resize([1500, h], Image.ANTIALIAS)
    else:
        if w <= 1500 and h > 150:
            im = im.resize([w, 150], Image.ANTIALIAS)

    if w > 1500 and h > 100:
        im = im.resize([1500, 150], Image.ANTIALIAS)

    a4im = Image.new('L',
                     (1500, 150),  # A4 at 72dpi
                     (255))  # White
    a4im.paste(im, im.getbbox())  # Not centered, top-left corner
    # img=a4im.filter(ImageFilter.GaussianBlur(1))
    # threshold = 200
    # img = a4im.point(lambda p: p > threshold and 255)
    img = a4im
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  ## shift for centering 0.0 (x,y)
    rows = np.size(img, 0)  # taking the size of the image
    cols = np.size(img, 1)
    crow, ccol = int(rows / 2), int(cols / 2)

    original = np.copy(fshift)
    n1 = int(crow - crow / 10)
    n2 = int(ccol - ccol / 10)
    fshift[crow - n1:crow + n1, ccol - n2:ccol + n2] = 0
    f_ishift = np.fft.ifftshift(original - fshift)
    img_back = np.fft.ifft2(f_ishift)  ## shift for centering 0.0 (x,y)
    img_back = np.abs(img_back)

    img_back = Image.fromarray(img_back)
    img_back=img_back.resize([512,48])
    img_back = img_back.convert('L')
    img_back.save(newPath)


#extract the IAM Handwriting Database from the sub folders, and processing the images for noise cleaning
dir='lines'
new_dir='datasets/sentences/trainB/'
create_large_folder(dir,new_dir)

"""
#creating a numpy array from the all images nd save it on npy file
file_list = prepare_data(new_dir)
resized_image_array=read_image_array_np_GRAYSCALE(file_list,600,32)
create_npy_file(resized_image_array,"x_data")
"""


