<p align="center">
  <img src=".github/readme/index.jpg" />
</p>

<br />

Implementing a keras model for diabetes recognition, using **Pima Indians** dataset.

## Model

Here is the model that we use to recognize diabetes, we create a sequential model with 3 hidden layers:

```python
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Fit

```python
EPOCHS = 150
BATCH_SIZE = 10
````

```python
# fit the keras model on the dataset
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
```
