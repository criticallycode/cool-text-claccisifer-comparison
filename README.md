This repo demonstrates how deep neural networks can be used to classify text data, much like they can be used to classify images. There are different methods for classifying text data, and this repo compares multiple methods. This repo demonstrates how to classify a small text dataset and a larger text dataset, using embeddings in the neural network. A Long Short-Term Memory neural network is also instituted for the sake of comparison. This repo comes with a Jupyter notebook where the classification can be carried out and the arguments of the classifiers tweaked.

The first section of the script classifies a small dataset, the Sentiment Labelled Dataset available from the UCI Machine Learning Repository. First, all necessary imports are made, the text data is converted to a CSV and concatenated, and then the training and testing data is created.

```Python
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Dense, GlobalMaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence

# Converts text to CSV
def csv_convert(input_data, data_source):
    # separate labels and sentences on tabs
    df = pd.read_csv(input_data, names=['sentence', 'label'], sep='\t')
    df['source'] = data_source
    return df

amz_data = "amazon_cells_labelled.txt"
imdb_data = "imdb_labelled.txt"
yelp_data = "yelp_labelled.txt"

amz_data = csv_convert(amz_data, "Amazon")
imdb_data = csv_convert(imdb_data, "IMDB")
yelp_data = csv_convert(yelp_data, "Yelp")

print(amz_data.head(5))

df_complete = pd.concat([amz_data, imdb_data, yelp_data])
df_complete.to_csv("sentiment_labelled_complete.csv")
print(amz_data.head(10))

# Separate out the features and labels from the CSV

features = df_complete['sentence'].values
labels = df_complete['label'].values

# Creating training/testing features and labels
# Still need to tokenize them
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=108)
```

Next, tokens are created for the word features, the sequence sizes are standardized, and a chosen size for the embedding is selected.

```Python
# Create the tokenizer, fit it on the text data
# use text_to_sequences to actually conver the features to tokens
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Need this for the creation of the network
vocab_len = len(tokenizer.word_index) + 1

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Selects size/complexity of the embeddings
embedding_dim = 50
```

The next section creates the classification model and fits the model.

```Python
# Creates the text classification model
def sentiment_model(embedding_dim):
    model = Sequential()
    # include the word embedding layers
    model.add(Embedding(vocab_len, embedding_dim, input_length=maxlen))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

model = sentiment_model(embedding_dim)

records = model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=10)
```

Now the model's performance is evaluated.

```Python
_, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
_, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'r', label='Training acc')
    plt.plot(x, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy Over EPochs')

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'r', label='Training loss')
    plt.plot(x, val_loss, 'b', label='Validation loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.show()

plot_history(records)
```

The script then implements a classifier on a second dataset, the IMDB dataset, which is larger than the first dataset. This loads in the data. (Thanks to Matthew Kerian/Cheez's [fix](https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56243777) for the data loading error.)

```Python
# Bypass bug in current version of Keras, allows use of imdb dataset
# modifies the default parameters of numpy's load function
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=10000)

# get features and labels, make sure they are the same number
features = np.concatenate((X_train, X_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
```

We now create the convolutional model for this dataset.

```Python
# Create the convolutional model with the embedding layer

def conv_model(max_words):

    model = Sequential()
    model.add(Embedding(10000, 64, input_length=max_words))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

model = conv_model(max_words)

records = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)
```

Then comes the evaluation.

```Python
# model evaluation
accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1]*100))
plot_history(records)
```

Finally, the LSTM model is created, fitted, and evaluated.

```Python
# Create the LSTM model

def LSTM_model(max_words):
    model = Sequential()
    model.add(Embedding(10000, 32, input_length=max_words))
    model.add(LSTM(64, dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(20))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model

model = LSTM_model(max_words)
records = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)

# model evaluation
accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1]*100))

plot_history(records)
```

![]()

