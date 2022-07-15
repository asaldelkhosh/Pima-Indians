# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# dataset path
PATH = './dataset/pima-indians-diabetes.csv'

# model configs
EPOCHS = 150
BATCH_SIZE = 10
ACCURACY_SCALE = 100


if __name__ == "__main__":
    # load the dataset
    dataset = loadtxt(PATH, delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    # fit the keras model on the dataset
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*ACCURACY_SCALE))
