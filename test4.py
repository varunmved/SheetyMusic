import tensorflow as tf
import numpy as np
import os
def adjust(filename):
  filename_queue = tf.train.string_input_producer([filename]) #  list of files to read

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
      image = my_img.eval(session=sess) #here is your image Tensor :) 

    print(image.shape)
    print(np.squeeze(np.asarray(image)))

    #Image.show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)


dir = 'data/classical-sheet-music'
l = []

for filename in os.listdir(dir):
  l.append('data/classical-sheet-music/' +  str(filename))

adjust(l[0])
'''
for files in l:
  adjust(files)
'''
