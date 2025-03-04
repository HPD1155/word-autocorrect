# AI Autocomplete
This is a simple implementation of an autocorrect engine in Python. It is made with Tensorflow and Keras. It is meant to autcorrect or "autocomplete" a word with only given a segment of it.<br>
Here is an example of how it works: <br>
```
input: hel
output: hello
```

# How it works
When creating this, I made the dataset by hand. I didn't want this to be a fully functional thing but rather a demo.
The dataset was created in a CSV format. It has a mistyped/partially typed word on the X or input side, and then the Y or output side has the completed or corrected word.
It will preprocess it with the custom preprocessor I made which split the word by character and tokenizes it. I did this because with traditional NLP tokenization methods
such as by word would not have worked as efficiently. This is because character by character tokenization helps the Network learn more complex and patterns. This will also improve
the accuracy overall. Also, mispelled words can be unique so tokenization by character would help solve the problem of the user entering something not in the vocabulary list. For the Y axis or output, I took all the unique labels and assigned them each a number.
Next I would make a Sequential model. It will have the obvious Embeddings layer. Then we will add two LSTM layers one being bidirectional. This will help the network recognize patters on the front and back of the data. It will help it get context on inputs in simple terms. Then the last layer is a Fully connected Dense layer with the `softmax` activation. What that would do is classify the textual input as one of the labeled words. We will train it wiith the Adam optimizer and Sparse_categorical_crossentropy. This is commonly
used for multiclass classification. When I run the model, it will take a misspelled word input from the user. It will then proceed to classify it. The classification is returned in an index where each index is one lablem and the label contains the probability of the prediction being said class. So find the label, we will get the index of the highest percentage in the array. That index will be converted to a label. The run function would proceed to return the label along with the confidence of it being said label.

