# Handwritten-Digit-Classification-Using-Convolutional-Neural-Networks-CNNs




# Project: Image Classification & Speech Recognition with CNNs and RNNs

## Overview
This project implements two models:
1. **Image Classification** using **Convolutional Neural Networks (CNNs)**.
2. **Speech Recognition** using **Recurrent Neural Networks (RNNs)** or **Long Short-Term Memory (LSTM)** networks.

Both models are trained to perform tasks like:
- Image classification from datasets like **MNIST** (handwritten digits) or **CIFAR-10** (common objects).
- Speech-to-text recognition for spoken commands or phrases.

## Training the Model

```python
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
