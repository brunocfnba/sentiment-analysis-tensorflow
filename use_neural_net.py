import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentiment_neural_net import nn_model
from sentiment_neural_net import train_neural_network
lemmatizer = WordNetLemmatizer()

x = tf.placeholder('float')


# function responsible to receive a sentence, prepare it (tokenizing, lemmatizing and tranforming into the hot vector
def use_neural_network(input_data):
    prediction = nn_model(x)
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = np.array(list(features))

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)
# Uncomment the row below to train the model
# train_neural_network(x)

# call the 'use_neural_network' providing a sentence to check the neural network return
use_neural_network("I really don't like that thing")
use_neural_network("This was the best store i've ever seen.")
use_neural_network("Why do you hate the world")
use_neural_network("we always need to do good things to help each other")
