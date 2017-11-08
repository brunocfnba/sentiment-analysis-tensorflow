import tensorflow as tf
import pickle

x = tf.placeholder('float')
y = tf.placeholder('float')

batch_size = 1000
num_epochs = 1


def load_details():
    with open('data_details.pkl', 'rb') as details:
        det = pickle.load(details)
        return det


line_sizes = load_details()


# Creates the neural network model
def ff_neural_net(input_data):
    neurons_hl1 = 1500
    neurons_hl2 = 1500
    neurons_hl3 = 1500

    output_neurons = 2

    l1_weight = tf.Variable(tf.random_normal([line_sizes['dict'], neurons_hl1]), name='w1')
    l1_bias = tf.Variable(tf.random_normal([neurons_hl1]), name='b1')

    l2_weight = tf.Variable(tf.random_normal([neurons_hl1, neurons_hl2]), name='w2')
    l2_bias = tf.Variable(tf.random_normal([neurons_hl2]), name='b2')

    l3_weight = tf.Variable(tf.random_normal([neurons_hl2, neurons_hl3]), name='w3')
    l3_bias = tf.Variable(tf.random_normal([neurons_hl3]), name='b3')

    output_weight = tf.Variable(tf.random_normal([neurons_hl3, output_neurons]), name='wo')
    output_bias = tf.Variable(tf.random_normal([output_neurons]), name='bo')

    l1 = tf.add(tf.matmul(input_data, l1_weight), l1_bias)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, l2_weight), l2_bias)
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, l3_weight), l3_bias)
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_weight) + output_bias

    return output


def training(in_placeholder):
    nn_output = ff_neural_net(in_placeholder)
    saver = tf.train.Saver()
    # We are using cross entropy to calculate the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_output, labels=y))

    # and Gradient Descent to reduce the cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # A TensorFLow session is created that will actually run the previously defined graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        for epoch in range(num_epochs):
            epoch_loss = 0
            buffer_train = []
            buffer_label = []
            with open('train_hot_vectors.pickle', 'rb') as train_hot_vec:
                for i in range(line_sizes['train']):
                    hot_vector_line = pickle.load(train_hot_vec)
                    buffer_train.append(hot_vector_line[0])
                    buffer_label.append(hot_vector_line[1])

                    # print('Bla:' + str(buffer_label))

                    if len(buffer_train) >= batch_size:
                        _, cost_iter = sess.run([optimizer, cost],
                                                feed_dict={in_placeholder: buffer_train, y: buffer_label})
                        epoch_loss += cost_iter
                        buffer_train = []
                        buffer_label = []

            print('Epoch {} completed. Total loss: {}'.format(epoch+1, epoch_loss))

        correct = tf.equal(tf.argmax(nn_output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        with open('test_hot_vectors.pickle', 'rb') as train_hot_vec:
            buffer_test = []
            buffer_test_label = []
            for i in range(line_sizes['test']):
                test_hot_vector_line = pickle.load(train_hot_vec)
                buffer_test.append(test_hot_vector_line[0])
                buffer_test_label.append(test_hot_vector_line[1])

        # the accuracy is the percentage of hits
        print('Accuracy using test dataset: {}'
              .format(accuracy.eval({in_placeholder: buffer_test, y: buffer_test_label})))
        # saver = tf.train.Saver()
        saver.save(sess, "model.ckpt")


# training(x)
