# AI Autocomplete
This is a simple implementation of an autocomplete engine in Python. It is made with Tensorflow and Keras. It is meant to autocomplete a word with only given a segment of it.<br>
Here is an example of how it works: <br>
```
input: hel
output: hello
```

# How it works
I first made a custom Preproccessing for the X label. It sorts out the input by character rather than word. This is so that the model can recognize more complex patterns. It would be very inefficent to preprocess by word. I then ran the characters into the keras tokenizer text to sequence to assign labels to each character. I then padded it so they all have the same dimensions. When dealing with the y or ouput, I just assigned each label with a number. I trained it with 2 LSTM layers and one being bidirectional. It returns with the softmax activation. The softmax activation will return a sequence of percents in the form of decimal. This represents the probability of the given label matching the input. <br>
I then make it determine the word by getting the max value of the probabilities. I then run the sequence of the max probability to translate into the word.<br>
Next I use matplotlib to show the probabilities of each label being the possible word.