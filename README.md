# Sentiment Analysis using TensorFlow
This is a basic neural network in TensorFlow with one input layer and two hidden layers to identify sentiment in sentences.
<BR><BR>The dataset used belongs to a Stanford project called Sentiment140 and can be downloaded [here](http://help.sentiment140.com/for-students/).
<BR><BR>Also this neural network is using basic techniques during the data preparation phase, feel free to improve that.
  
### How to run the code
After downloading the repo, open the 'use_neural_net.py' file and uncomment line 46 so your neural network can be trained. Once trained use the 'use_neural_network' function providing a sentence as parameter to get its sentiment.

### Files description
- **data_prep.py** - contains all the functions responsible for preparing all the sample sentences and let them ready to be fed into the neural network.
- **sentiment_neural_net.py** - where the neural netowrk model is created, all the training happens and the model is saved.
- **use_neural_net.py** - kick off code used to call the training function as well as use the neural network.
