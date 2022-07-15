# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# dataset path
PATH = './dataset/pima-indians-diabetes.csv'


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
    model.fit(X, y, epochs=150, batch_size=10)
    