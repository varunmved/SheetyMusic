import tensorflow as tf
import numpy as np
import os
from PIL import Image
#import PIL.Image

filesList = []
dir = 'data/classical-sheet-music'

for filename in os.listdir(dir):
  filesList.append('data/classical-sheet-music/' +  str(filename))



#don't really know what this was for

'''
print(filesList)

filename_queue = tf.train.string_input_producer(filesList)
reader = tf.WholeFileReader()

#key,value = reader.read(filename_queue)

#print(key,value)
'''

def adjust():
  filename_queue = tf.train.string_input_producer(filesList) #  list of files to read

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(len(filesList)): #length of your filename list
      image = my_img.eval(session=sess) #here is your image Tensor :) 
      l = (np.ndarray.flatten(image))
    #print(np.squeeze(np.asarray(image)[0))
    #print(l)
    #np.savetxt('out.txt',l) 
  
    #print(image[:,:,0])
    #print(np.average(image[:,1]))
    #a = np.squeeze(image[:,2])
    #print(a)
    #print(np.squeeze(np.asarray(image)[0))

    #Image.show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)
    return l

out = adjust()
outnp = np.ndarray.reshape(out,(1181,835,3))
my_img = tf.image.encode_png(outnp) # use png or jpg decoder based on your files.
#Image.show(Image.fromarray(np.asarray(my_img)))
img = PIL.Image.show(Image.fromarray(np.asarray(my_img)))
img.show()
'''
#comment out to loop through everything
for files in l:
  adjust(files)
'''
