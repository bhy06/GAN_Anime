# -*- coding: utf-8 -*-
import cv2
import os.path
from PIL import Image

# im from PIL.Image.open, face margin
def faceCrop(im,x,y,w,h,m):
    """
    m is the relative margin added to the face image
    """
    sizeX, sizeY = im.size
    new_x, new_y = max(0,x-m*w), max(0,y-m*h)
    new_w = w + 2*m*w if sizeX > (new_x + w + 2*m*w) else sizeX - new_x
    new_h = h + 2*m*h if sizeY > (new_y + h + 2*m*h) else sizeY - new_y
    new_x,new_y,new_w,new_h = int(new_x),int(new_y),int(new_w),int(new_h)
    return im.crop((new_x,new_y,new_x+new_w,new_y+new_h))


def detect(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    
    if len(faces)==0:
        return 0
    
    im = Image.open(filename)   
    im = faceCrop(im, faces[0][0], faces[0][1], faces[0][2], faces[0][3], 0.25) 
    im = im.resize((96, 96), Image.ANTIALIAS)
    
    return im   
    
def buildFolder(path):
    for f in os.listdir(path):
        new_dir_path = os.path.join('faces',f)
        os.makedirs(new_dir_path)

base_dir = "gallery-dl\\danbooru"

# build faces folder
buildFolder(base_dir)

cls_dirs = [f for f in os.listdir(base_dir)]
imgs = []
for i in range(len(cls_dirs)):
    sub_dir = os.path.join(base_dir, cls_dirs[i])
    imgs_tmp = [os.path.join(sub_dir,f) for f in os.listdir(sub_dir) if f.endswith(('.jpg', '.png'))]
    imgs = imgs + imgs_tmp
    
# detect and save new images
count = 0
for img in imgs:
    tmp = img.split('\\')
    cls_name,img_name = tmp[len(tmp)-2], tmp[len(tmp)-1]
    new_dir_path = os.path.join('faces',cls_name)

    new_img_path = os.path.join(new_dir_path, img_name)
    
    img = detect(img)
    
    if img != 0:
        img.save(new_img_path)
    
    count = count + 1
    if count % 10000 == 0:
        print(count)
    
    
    
    
    
    
    
    
    
