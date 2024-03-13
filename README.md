# goodread_prediction
My main task was to predict the rating of a random book, which I achieved by implementing the appropriate machine learning algorithms. I built three models: a Baseline Machine Learning Model where I used a Logistic Regression model, a 3-layer Neural Network Model, and a Deep Neural Network (DNN). For the Complex Neural Network, I used Long Short-Term Memory (LSTMs), which is a type of Recurrent Neural Network (RNN), Gated Recurrent Units (GRUs) which are again a type of Recurrent Neural Networks (RNN) used for modeling sequential data, and Bidirectional Encoder Representations from Transformers (BERT). BERT uses a bidirectional approach to generate contextualized word embeddings, which are representations of words that capture their meanings in the context of a sentence. This approach is beneficial because it is pre-trained on massive amounts of text data, allowing it to learn general language patterns and relationships between words.

# Clean review_text

I cleaned the review text and removed the noise caused by abundant punctuations, wide spaces, and invalid characters, which would otherwise have been responsible for the inefficiency in executing the algorithms I used. It's vital to get rid of unnecessary input values and classify nouns, pronouns, and verbs.

I split the data into training, testing, and validation according to the best fit and kept it consistent to avoid void/inefficient results and achieve optimum efficiency.

# Model

## Standard ML Baseline: Logistic regression

For my standard machine learning baseline model, I chose to use the Logistic Regression Model. It's a widely used statistical method for binary classification tasks, where the goal is to predict a binary outcome, such as whether a book is highly rated or not. I believe it can be a useful method for classification tasks, and its performance can be easily assessed using performance metrics like the F1 Score, among others.

## 3Layer NN Baseline:

My 3-Layer Neural Network Model consists of an input layer, a hidden layer, and an output layer, where each layer contains one or more neurons. These neurons are computational units that perform mathematical operations on the input data. Given the necessity to experiment with different parameters for the task at hand, I find this model to be the best fit for such tasks.

## Deep NN

For my deep neural networks model, I decided to create a 7-layer DNN model that includes 3 base layers of neurons. I explored the optimum number of layers needed and concluded that adding more layers would incrementally improve performance. Based on my best result, which was achieved with 15 layers and 55 neurons (the best configuration from the initial 3 layers), I tested various activation functions like relu, sigmoid, tanh, and elu. I did this to gain a better understanding of the model and to enhance its performance.

## Complex Neural Network Models:

I have prominently used 3 Complex Neural Network Models namely, Recurrent Neural Network Model (RNN), Long Short-Term Models (LSTMs) and Bidirectional Encoder Representations from Transformer (BERT).

## RNN

Recurrent Neural Network Model (RNN): it is a type of neural network designed to handle sequential data as against the neural networks which process data in a fixed sequence of layers. I found it difficult to find the parameters to tune the model however, I was successful in generating a decent output for the same.

## GRU

I've been working with Gated Recurrent Units (GRUs), which are essentially a variant of Recurrent Neural Networks designed to handle sequential data, like text and speech. They're similar to LSTMs because they also use gating mechanisms to control the flow of information through the network. However, in my experience, I've noticed that the performance of the model depends on several factors, such as the quantity/size of the input data and the choice of parameters/hyperparameters, among others.

## BERT

I used the Bidirectional Encoder Representations from Transformer (BERT), a pre-trained deep learning model, because I found it incredibly effective for language processing tasks. I chose it as I believe it's a powerful tool for handling large amounts of data. However, I also realized that it has a significant drawback: it requires substantial computational resources to train and enhance the model's performance.

# Result

After performing the given task, I clearly saw that the Bidirectional Encoder Representations from Transformer Model (BERT) outperformed all the other models I used. I would highly recommend using it because of its superior ability to understand the contextual meaning of words in a sentence and its efficiency in handling large datasets.

The models I would not recommend are the Recurrent Neural Networks Model (RNN) and Deep Neural Networks Model (DNN) due to their inefficient performance and their inability to effectively process my dataset.

The mean and standard deviation of all the models I tested are 0.331 and 0.227, respectively. Through my approach and learnings, I discovered the importance of a model's efficiency in managing sequential datasets that are large in volume. Additionally, it's crucial that the model executes the required task well and predicts outcomes accurately.

I experimented with various models, including the Logistic Regression Model, 3 Layers Neural Networks Model (CNN), Recurrent Neural Networks Model (RNN), Deep Neural Networks Model (DNN), Gated Recurrent Units (GRUs), and Bidirectional Encoder Representation from Transformer Model (BERT). Among these, I found that adjusting the number of epochs settings was challenging. I couldn't run the models as efficiently as I wanted due to Random Access Memory (RAM) and user interface limitations, particularly in Google Colab. Furthermore, I learned that increasing the number of epochs tended to improve the accuracy score. However, due to the mentioned limitations, I was unable to execute the models at the desired level of epochs, which prevented me from achieving optimum accuracy.

# Summary

In my analysis, the table I reviewed showcases the model and its respective accuracy, leading me to conclude that the Bidirectional Encoder Representations from Transformers (BERT) stands out. I can explain this as follows:

Firstly, BERT is incredibly adept at grasping the contextual meanings of words in sentences, from which it meticulously extracts useful features. Secondly, its capacity to process vast amounts of data allows me to efficiently train the dataset with BERT. Lastly, because BERT is pre-trained on extensive data collections, it effortlessly learns language patterns, which I can then fine-tune to perform any specific classification task.

After uploading the file on Kaggle, I achieved a score of 0.5808, which is decent considering I only set the training model to run for 2 epochs. However, I believe this could be significantly improved by increasing the number of epochs. This would likely enhance the accuracy score further. Therefore, I've observed that the number of epochs is directly proportional to the accuracy score. To achieve better results in the task, setting the epochs to a higher number, like 50, could potentially increase the score by approximately 20%.
