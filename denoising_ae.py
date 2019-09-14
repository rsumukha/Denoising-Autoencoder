import os
import gzip
import numpy as np
import matplotlib.pyplot as plt


def fashion_mnist(path):
    # Load training data
    path = ''
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    # Load Test Data
    path = ''
    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)
    return trData, trLabels, tsData, tsLabels


def sigmoid(Z):
    '''
    computes sigmoid activation of Z
    '''
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, Z):
    '''
    computes derivative of sigmoid activation
    '''
    # print("dA shape and Z shape",dA.shape)
    sig, cache = sigmoid(Z)
    dZ = dA * sig * (1.0 - sig)
    return dZ


def cost_estimate(A2, X):
    '''
    '''
    ### CODE HERE
    epsilon = 0.001
    cross_entropy = - np.mean(np.sum(X * np.log(A2 + epsilon) + (1 - X) * np.log(1 - A2 + epsilon), axis=1))
    return cross_entropy


class DenoisingAutoencoder:
    def __init__(self, n_in, n_h):
        '''Initialise Denoising Autoencoder'''
        self.W = np.random.randn(n_h, n_in)
        self.b0 = np.random.randn(n_h, 1)
        self.b1 = np.random.randn(n_in, 1)

    def get_corrupt_data(self, input, corrupt_level):
        '''This function introduce noise in input data'''
        assert corrupt_level < 1
        input_curr = np.random.binomial(size=input.shape, n=1, p=1 - corrupt_level) * input

        # print("Shape of currpt input ",input_curr.shape)
        return input_curr

    def encode(self, X):
        '''This function maps the input data onto the hidden layer (encoding)'''
        Z = np.dot(self.W, X) + self.b0
        A, cache = sigmoid(Z)
        return A, cache

    def decode(self, A1):
        '''This function map the hidden layer onto the output layer (decoding)'''
        # using tied weights
        Z = np.dot(self.W.T, A1) + self.b1
        A, cache = sigmoid(Z)
        return A, cache

    def train(self, X, epochs, learning_rate, corrupt_level):
        '''This function trains the model for given input'''
        for epoch in range(0, epochs):
            if corrupt_level!=0:
                X_curr = self.get_corrupt_data(X, corrupt_level)
            encoded, cache1 = self.encode(X_curr)
            decoded, cache2 = self.decode(encoded)

            # print("Shape of X and X_decoded", X.shape,decoded.shape)

            L_h2 = X - decoded
            dZ2 = sigmoid_der(L_h2, cache2["Z"])
            db1 = np.sum(dZ2, axis=1, keepdims=True)
            dW1 = np.dot(dZ2, encoded.T)

            L_h1 = np.dot(self.W, dZ2)
            dZ1 = sigmoid_der(L_h1, cache1["Z"])
            db0 = np.sum(dZ1, axis=1, keepdims=True)
            dW0 = np.dot(dZ1, X.T)

            dW = dW1.T + dW0

            self.W += learning_rate * dW
            self.b0 += learning_rate * db0
            self.b1 += learning_rate * db1

            if epoch % 10 == 0:
                print("Epoch no. ", epoch)
                # cross_entropy = cost_estimate(decoded, X)
                # print("cross entropy ",cross_entropy)
            if epoch % 100 == 0:
                cross_entropy = cost_estimate(decoded, X)
                print("cross entropy ", cross_entropy)

    def reconstruct(self, input):
        '''This function reconstructs data from corrupted data'''
        encoded, _ = self.encode(input)
        decoded, _ = self.decode(encoded)
        return decoded




def main():
    X_train, Y_train, X_test, Y_test = fashion_mnist('Dataset/train')
    # print("Shape of X and Y train ",X_train.shape, Y_train.shape)
    stacked_autoencoder(X_train, Y_train, X_test, Y_test)
    exit(0)
    print("Shape of X and Y test ", X_test.shape, Y_test.shape)

    n_in = 28 * 28
    n_h = 100

    DA = DenoisingAutoencoder(n_in, n_h)
    DA.train(X_train.T, epochs=10, learning_rate=0.1, corrupt_level=0.1)

    # polt original image from test images
    print("Original Image")
    org_image = X_test[0].reshape([28, 28])
    plt.gray()
    plt.imshow(org_image)
    plt.show()

    # corrupt test images and plot one of them
    print("Corrupt Image")
    corrypt_test_images = DA.get_corrupt_data(X_test.T, corrupt_level=0.1)
    plt.gray()
    plt.imshow(corrypt_test_images.T[0].reshape([28, 28]))
    plt.show()

    # reconstruct test images and plot it
    print("Reconstructed Image")
    test_pred = DA.reconstruct(corrypt_test_images)
    test_pred = test_pred.T
    plt.gray()
    plt.imshow(test_pred[0].reshape([28, 28]))
    plt.show()

    # X = X_train[:].reshape([28, 28]);
    # plt.gray()
    # plt.imshow(X_train)
    # plt.show()
    return

if __name__ == '__main__':
    main()
