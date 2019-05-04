from numpy import array
from sklearn.preprocessing import LabelEncoder
import re
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences

def ezip(ls1, ls2):
    count = 0
    e = [] 
    for index in range(len(ls1)):
        e.append([count, ls1[index], ls2[index]])
        count += 1
    return e


def readfile(file):
    input = []
    with open(file, encoding = "utf-8 ") as f:
        test = 0
        for line in f:
                input.append(line.strip())
    return input

print("Reading files.")
train_input = readfile("train.txt")
test_input = tuple(readfile("test.txt"))
words = readfile("words.txt")
words = tuple([tuple(k.split(',')) for k in words])

print("Formatting test set.")
pattern = re.compile(' ?\{([^\|]+)\|([^\}]+)\} ?')
X_tests = []
for couple in words:
    X = []
    for index, phrase in enumerate(test_input):
        m = re.search(pattern, phrase)
        if couple[0] in [m.group(1), m.group(2)] or couple[1] in [m.group(1), m.group(2)]:
            X.append(phrase[:phrase.find('{')])
    X_tests.append(X)

def format_trains(i):
    print("Formatting train set.")
    X_trains = []
    y_trains = []
    for couple in words:
        X = []
        y = []
        for phrase in i:
            if couple[0] in phrase.split(' '):
                X.append(phrase)
                y.append(couple[0])
            elif couple[1] in phrase.split(' '):
                X.append(phrase)
                y.append(couple[1])
        X_trains.append(X)
        y_trains.append(y)
    return X_trains, y_trains

def array_average_in_place(lst):
    result = []
    for row in range(len(lst[0])):
        temp = []
        sum = 0
        for array in lst:
            #sum += array.item(row, 0)
            temp.append(array.item(row, 0))
        result.append(temp)
        #result.append(sum/len(lst))
    print(len(result))
    return result

X_trains, y_trains = format_trains(train_input)
output = []

tokenizer = Tokenizer()
encoder = LabelEncoder()

for set in range(len(X_trains)):
    tokenizer.fit_on_texts(X_trains[set])
    sequences = tokenizer.texts_to_sequences(X_trains[set])
    X = array(sequences)
    maxlength = max(map(len, X))
    X = pad_sequences(X, maxlen = maxlength)

    vocab_size = len(tokenizer.word_index) + 1

    y = encoder.fit_transform(y_trains[set])
    y = to_categorical(y, num_classes = 2)
    seq_length = maxlength

    model = Sequential()
    model.add(Embedding(vocab_size, 2, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X, y, batch_size = 128, epochs = 1)

    X_test = tokenizer.texts_to_sequences(X_tests[set])
    X_test = array(X_test)
    X_test = pad_sequences(X_test, maxlen = maxlength)

    prediction = model.predict(X_test)
    print(prediction)
    output += [k[0] for k in prediction]


with open('output.csv', 'w+') as file:
    for x in outfmt:
        file.write(x + '\n')