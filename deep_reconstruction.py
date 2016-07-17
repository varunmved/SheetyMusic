""" use a multilayer perceptron to learn the patterns in a few pieces of sheet music
and then have it generate a new piece of sheet music
    goals:
        1. get use to the api for deep models
        2. create something better than grey blocks as seen in the linear model
"""

from PIL import Image
import tensorflow as tf
import numpy as np
import traceback
import pdb


# our test images to train the model
filenames = ["data/example_sheet.png"]
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
k, v = reader.read(filename_queue)
sample_img = tf.image.decode_png(v)


# model parameters for the multilayer perceptron
learning_rate = 0.001
training_epochs = 10
display_step = 1

n_hidden_1 = 128     # 1st layer number of features
n_hidden_2 = 128     # 2nd layer number of features
n_hidden_3 = 128     # 3rd layer number of features
n_inputs = 2         # input is the x,y coordinate of a pixel in an image
n_outputs = 1       # one output, the regressed value. also the number of nodes in final layer

n_pixels = 986135   # 835 x 1181 (W x H)
img_width = 835.0
img_height = 1181.0

# tf Graph input
X = tf.placeholder(tf.float32, [n_pixels, n_inputs])
y = tf.placeholder(tf.float32, [n_pixels, n_outputs])

# weights and biases for all the nodes in each layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_outputs]))
    }
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'out': tf.Variable(tf.zeros([n_outputs]))
    }


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer 1 with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer 2 with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer 2 with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


def main():
    # define the model
    pred = multilayer_perceptron(X, weights, biases)

    # cost function and optimization function
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_pixels)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(len(filenames)):
            image = sample_img.eval()

        original = Image.fromarray(np.asarray(image))
        greyscale_img = np.apply_along_axis(np.mean, 2, original)
        train_X = np.ndarray([n_pixels, n_inputs]) 
        train_Y = np.ndarray([n_pixels, n_outputs]) 
        
        for epoch in range(training_epochs):
            total_cost = 0
            for w in range(0, int(img_width)):
                for h in range(0, int(img_height)):
                    sample_X = np.reshape(
                            np.array([w / img_width, h / img_height]),
                            [1, n_inputs]
                            )
                    sample_Y = np.reshape(np.array([greyscale_img[h][w] / 255.0]), [1, 1])
                    train_X[w * int(img_width) + h] = sample_X
                    train_Y[w * int(img_width) + h] = sample_Y
                    # _, c = sess.run([optimizer, cost], feed_dict={X: sample_X, y: sample_Y})
                    # print("%s, %s -- %f" % (w, h, c))
                    # total_cost += c

            _, total_cost = sess.run([optimizer, cost], feed_dict={X: train_X, y: train_Y})
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(total_cost))
        
        pdb.set_trace()
        print("Optimization Finished!")
        

if __name__ == "__main__":
    main()
