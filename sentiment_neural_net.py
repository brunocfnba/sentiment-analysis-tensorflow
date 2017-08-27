import tensorflow as tf
import pickle
import numpy as np

x = tf.placeholder('float')
y = tf.placeholder('float')

nodes_hl1 = 1500
nodes_hl2 = 1500
nodes_hl3 = 1500

n_classes = 2
batch_size = 100
epochs = 10

# opens the pickle file with hot vectors
with open('sentiment_set.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)


# Create all the weights and biases TensorFlow variables for each layer
layer1 = {'weight': tf.Variable(tf.random_normal([len(train_x[0]), nodes_hl1]), name='w1'),
          'bias': tf.Variable(tf.random_normal([nodes_hl1]), name='b1')}

layer2 = {'weight': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2]), name='w2'),
          'bias': tf.Variable(tf.random_normal([nodes_hl2]), name='b2')}

layer3 = {'weight': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3]), name='w3'),
          'bias': tf.Variable(tf.random_normal([nodes_hl3]), name='b3')}

output_layer = {'weight': tf.Variable(tf.random_normal([nodes_hl3, n_classes]), name='wo'),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


# Actually creates the neural network model
def nn_model(input_data):

    l1 = tf.add(tf.matmul(input_data, layer1['weight']), layer1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, layer2['weight']), layer2['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, layer3['weight']), layer3['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = nn_model(x)
    # We are using cross entropy to calculate the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # and Gradient Descent to reduce the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # A TensorFLow session is created that will actually run the previously defined graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0

            # this where we send chunks of data to be trained against our model
            while i < len(train_x):
                start = i
                end = i+batch_size
                chunk_x = np.array(train_x[start:end])
                chunk_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: chunk_x, y: chunk_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # the accuracy is the percentage of hits
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        saver = tf.train.Saver()
        saver.save(sess, "model.ckpt")
