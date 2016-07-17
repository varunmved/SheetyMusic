from PIL import Image
import tensorflow as tf
import numpy as np
import pdb
import traceback


filename_queue = tf.train.string_input_producer(['data/example_sheet.png']) #  list of files to read
reader = tf.WholeFileReader()

k, v = reader.read(filename_queue)
my_img = tf.image.decode_png(v)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1): #length of your filename list
        image = my_img.eval() #here is your image Tensor :) 
    
    # the original image with all color channels
    original = Image.fromarray(np.asarray(image))
    original.show()
    
    pdb.set_trace()

    # take the average of all 3 color channels to flatten the color dimension
    greyscale_img = np.apply_along_axis(np.mean, 2, original)

    # convert greyscale to bw. 
    greyscale_img[greyscale_img >= 128] = 255
    greyscale_img[greyscale_img < 128] = 0
    new = Image.fromarray(greyscale_img)
    new.show()
    
    coord.request_stop()
    coord.join(threads)

