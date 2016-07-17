'''
encoding and decoding
used from:
https://jeremykarnowski.wordpress.com/2016/06/13/inputting-image-data-into-tensorflow-for-unsupervised-deep-learning/
'''


import numpy as np
# import cv2
from PIL import Image
import gzip
import os
import struct 
 
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('&amp;gt;')
  return np.frombuffer(bytestream.read(4), dtype=dt)
 
def _read8(bytestream):
  dt = np.dtype(np.uint8)
  return np.frombuffer(bytestream.read(1), dtype=dt)
 
def decode(imgf, outf, n, imgs=False):
  """
  Given a binary file, convert information into a csv or a directory of images
 
  If imgs is true, outf is a directory that will hold images
  directory MUST exist before this
  If imgs is false, outf will be a csv file
  """
  with gzip.open(imgf) as bytestream:
 
    if imgs:
      savedirectory = outf
    else:
      o = open(outf, "w")
 
    magic_num = _read32(bytestream)[0]
    num_images = _read32(bytestream)[0]
    num_rows = _read32(bytestream)[0]
    num_cols = _read32(bytestream)[0]
 
    images = []
 
    for i in range(n):
      image = []
      for j in range(num_rows * num_cols):
        image.append(_read8(bytestream)[0])
      images.append(image)
 
    if imgs:
      for j,image in enumerate(images):
        saveimage = np.array(image).reshape((num_rows,num_cols))
        result = Image.fromarray(saveimage.astype(np.uint8))
        result.save(outf + str(j) + '.png')
        # cv2.imwrite(outf + str(j) + '.png', np.array(image).reshape((num_rows,num_cols)))
    else:
      for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
      o.close()

def encode(imgd, imgf, ext='.png'):
  """
  Given a set of black white images (that probably have 3 channels), convert
  it into a gzipped file that has all the information that is in standard
  MNIST files
 
  imgd = give the folder extension. Must have the trailing '/' in the string
  imgf = the filename for saving
  Provide the extension of the images
  """
  fs = [imgd + x for x in np.sort(os.listdir(imgd)) if ext in x]
  num_imgs = len(fs)
  o = open(imgf, "w")
 
  # Write items in the header
  # Magic Number for train/test images
  o.write(struct.pack('>i', 2051))
  # Number of images
  o.write(struct.pack('>i', num_imgs))
 
  # Load the first image to get dimensions
  im = np.asarray(Image.open(fs[0]).convert('L'), dtype=np.uint32)
  # im = cv2.imread(fs[0])[:,:,0] # images must be one dimensional grayscale
  r,c = im.shape 
 
  # Write the rest of the header
  o.write(struct.pack('>i', r)) # Number of rows in 1 image
  o.write(struct.pack('>i', c)) # Number of columns in 1 image
 
  # For each image, record the pixel values in the binary file
  for img in range(num_imgs):
    # For opencv, images must be one dimensional grayscale
    # im = cv2.imread(fs[img])[:,:,0]
     im = np.asarray(Image.open(fs[img]).convert('L'), dtype=np.uint32)
     for i in xrange(im.shape[0]):
        for j in xrange(im.shape[1]):
           o.write(struct.pack('>B', im[i,j]))
 
  # Close the file
  o.close()
 
  # Gzip the file (as this is used in encoding)
  f_in = open(imgf)
  f_out = gzip.open(imgf + '.gz', 'wb')
  f_out.writelines(f_in)
  f_out.close()
  f_in.close()
  os.remove(imgf)


