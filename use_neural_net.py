import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentiment_neural_net import ff_neural_net
from sentiment_neural_net import training
lemm = WordNetLemmatizer()

x = tf.placeholder('float')


# function responsible to receive a sentence, prepare it (tokenizing, lemmatizing and tranforming into the hot vector
def get_sentiment(input_data):
    tf.reset_default_graph()
    pl = tf.placeholder('float')
    nn_output = ff_neural_net(pl)
    saver = tf.train.Saver()
    with open('word_dict.pickle', 'rb') as f:
        word_dict = pickle.load(f)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        words = word_tokenize(input_data.lower())
        lemm_words = [lemm.lemmatize(w) for w in words]
        hot_vector = np.zeros(len(word_dict))

        for word in lemm_words:
            if word.lower() in word_dict:
                index_value = word_dict.index(word.lower())
                hot_vector[index_value] += 1

        hot_vector = np.array(list(hot_vector))

        result = (sess.run(tf.argmax(nn_output.eval(feed_dict={pl: [hot_vector]}), 1)))
        # print(result)
        if result[0] == 0:
            print('Negative:', input_data)
        elif result[0] == 1:
            print('Positive:', input_data)


# Uncomment the row below to train the model
# training(x)

# call the 'use_neural_network' providing a sentence to check the neural network return
get_sentiment('Lebron is a beast... nobody in the NBA comes even close')
get_sentiment("This was the best store i've ever seen.")
get_sentiment("Why do you hate the world")
get_sentiment("we always need to do good things to help each other")

