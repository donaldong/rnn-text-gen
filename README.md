# Introduction
Text generation is a popular problem in Data Science and Machine Learning, and
it is a suitable task for Recurrent Neural Nets.  This report uses tensorflow
to build an RNN text generator and builds a high-level API in Python3. The
report is inspired by @karpathy  (min-char-rnn) and Géron (Hands-On Machine
Learning with Scikit-Learn and TensorFlow). This is a class project in
CST463 - Advanced Machine Learning at Cal State Monterey Bay, instructed by Dr.
Glenn Bruns.

# Modules
`Dataset`, `RNNTextGenerator`, and `ModelSelector` are the three main modules.

## Dataset
Defined in
[src/dataset.py](https://github.com/donaldong/rnn-text-gen/blob/master/src/dataset.py)

Creates a text dataset contains the one-hot encoded text data. It produces
batches of sequences of encoded labels. We split the text data into batches are
used to train the RNN, and we sample a random chuck of the text (with given
length) to evaluate the performance of our data.

### Attributes

#### seq_length
The number of consecutive characters in a slice of the text data (for batching).

#### vocab_size
The number of unique characters in the text data.

### Methods

#### constructor
```
Dataset(
  filenames,
  seq_length,
  shuffle=True
)
```
##### Args
- `filenames`
A list of filenames. They are the paths to one or more plain text files. The
file contents are concatenated in the given order.
- `seq_length`
The number of encoded labels in a sequence. It's the one-hot encoded output of
a slice of consecutive characters in the text.
- `shuffle`
Whether to shuffle the sequences. Default to `True`. When it is set to `False`,
it will batch the sequences in order of the original text.

#### encode
```
encode(text)
```
One-hot encode the text.
##### Args
- `text`
The original character sequence.
##### Returns
The one-hot encoded character sequence.
##### Example
```
dataset = Dataset([filename], seq_length)
encoded = dataset.encode(text)
assert len(encoded) == len(text)
for label in encoded:
    assert sum(label) == 1
    assert len(label) == dataset.vocab_size
```

#### decode
Decode the one-hot encoded sequence to text format.
```
decode(seq)
```
##### Args
- `seq`
The one-hot encoded character sequence.
##### Returns
The original character sequence.
##### Example
```
dataset = Dataset([filename], seq_length)
assert dataset.decode(dataset.encode(text)) == text
```

#### batch
```
batch(
    batch_size,
    drop_remainder=True
)
```
Produce many `batch`es. A `batch` has many input and target sequences (`inputs`
and `targets`). Each input and target sequence is a list of encoded labels,
and they offset by 1 timestep, thus they have the same length. With a
sequence `[l0, l0, l1, l1, l2]`, the input sequence would be `[l0, l0,
l1, l1]`, and the target sequence would be `[l0, l1, l1, l2]`. 
##### Args
- `batch_size`
The number of instances (sequences) in a single batch.
- `drop_remainder`
Whether the last batch should be dropped in the case of having fewer than
`batch_size` elements.
##### Returns
A number of batches which covers the text data. 
##### Example
```
dataset = Dataset([filename], seq_length)
for batch in dataset.batch(batch_size):
    # The number of elements in the batch is `batch_size`
    assert len(batch.inputs) == batch_size
    assert len(batch.targets) == batch_size
    for i in range(batch_size):
        # Each element in the batch is a sequence
        assert len(batch.inputs[i]) == seq_length
        assert len(batch.targets[i]) == seq_length
        for j in range(seq_length):
            # One-hot encoded
            assert sum(batch.inputs[i][j]) == 1
            assert len(batch.inputs[i][j]) == dataset.vocab_size
```

#### sample
```
sample(batch_size)
```
Radomly select some sequences (with replacement).
##### Args
- `batch_size`
The number of instances (sequences) in a single batch.
##### Returns
A single batch.
##### Example
```
dataset = Dataset([filename], seq_length)
count = 0
batch = dataset.sample(batch_size)
for seq in batch.inputs:
    assert len(seq) == seq_length
    for i in range(seq_length):
        # One-hot encoded
        assert sum(seq[i]) == 1
        assert len(seq[i]) == dataset.vocab_size
    count += 1
assert count == batch_size
```
