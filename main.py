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
