# Sentiment Analysis using TensorFlow
This is a basic neural network in TensorFlow with one input layer and two hidden layers to identify sentiment in sentences.
<BR><BR>There are 2 txt files (pos.txt and neg.txt) with around 5000 sample sentences on each. The after training this neural network with this sample you'll probably get an accuracy around 60% this is due to the small amount of data samples since a neural network works better the more samples is provided during the training phase.
<BR><BR>Also this neural network is using basic techniques during the data preparation phase, feel free to improve that.
  
### How to run the code
After downloading the repo, open the 'use_neural_net.py' file and uncomment line 40 so your neural network can be trained. Once trained use the 'use_neural_network' function providing a sentence as parameter to get its sentiment.

### Files description
- **pos.txt and neg.txt** - files with the sample sentences to train the neural network.
- **data_prep.py** - containes all the functions responsible for preparing all the sample sentences and let them ready for the neural network consume.
- **sentiment_neural_net.py** - the place where the neural netowrk model is defined, all the training happens and the model is saved by TensorFlow.
- **use_neural_net.py** - kick off code used to call the training function as well as use the neural network.
