""" fit a linear regression model to a pixel vector image representation and then use 
the linear model to reconstruct the image.
    goals:
        1. get familiar with tensorflow
        2. see what happens with a linear model
"""

from PIL import Image
import tensorflow as tf
import numpy as np
import traceback
import pdb

filename_queue = tf.train.string_input_producer(['data/example_sheet.png'])
reader = tf.WholeFileReader()
k, v = reader.read(filename_queue)
my_img = tf.image.decode_png(v)


# configure our linear regression model
learning_rate = 0.01
training_epochs = 100
display_step = 10

n_samples = 986135
X = tf.placeholder(tf.float32, shape=[n_samples, 5], name="features")
y = tf.placeholder(tf.float32, shape=[n_samples, 1], name="regressor")

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0.0, name="bias")

pred = tf.add(tf.matmul(X, W), b)
cost = cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    
    # load the image data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1): #length of your filename list
        image = my_img.eval() #here is your image Tensor :) 

    original = Image.fromarray(np.asarray(image))
    greyscale_img = np.apply_along_axis(np.mean, 2, original)
    
    train_X = np.ndarray([n_samples, 5])
    train_Y = np.ndarray([n_samples, 1])
    for epoch in range(training_epochs):
        for w in range(0, 835):
            for h in range(0, 1181):
                sample_X = np.reshape(np.array([h / 1181.0, w / 835.0, (h / 1181.0) * (w / 835.0),
                    (w / 835.0)**2,
                    (h / 1181.0)**2
                    ]), [1, 5])
                sample_Y = np.reshape(np.array([greyscale_img[h][w] / 255.0]), [1, 1])
                
                # sess.run(optimizer, feed_dict={X: sample_X, y: sample_Y})
                train_X[w * 835 + h] = sample_X
                train_Y[w * 835 + h] = sample_Y

        sess.run(optimizer, feed_dict={X: train_X, y: train_Y})

        #Display logs per epoch step
        print("epoch %s, optimized" % epoch)
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    reconstructed_Y = tf.add(tf.matmul(train_X.astype(np.float32), sess.run(W)), sess.run(b)).eval()
    foo = np.reshape(reconstructed_Y * 255, [1181, 835])
    Image.fromarray(np.asarray(foo)).show()
    original.show()


