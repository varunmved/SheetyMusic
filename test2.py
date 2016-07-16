import imageflow

convert_images(images, labels, filename)

# Distorted images for training
images, labels = distorted_inputs(filename='data/example.py', batch_size=FLAGS.batch_size,
                                      num_epochs=FLAGS.num_epochs,
                                      num_threads=5, imshape=[32, 32, 3], imsize=32)

'''
# Normal images for validation
val_images, val_labels = inputs(filename='../my_data_raw/validation.tfrecords', batch_size=FLAGS.batch_size,
                                    num_epochs=FLAGS.num_epochs,
                                    num_threads=5, imshape=[32, 32, 3])
'''
