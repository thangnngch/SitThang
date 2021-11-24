import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

dataHeart = pd.read_csv("/content/gdriver/MyDrive/heart.csv")
dataHeart = dataHeart.drop_duplicates(keep='first')
from sklearn.preprocessing import LabelEncoder

li = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for i in li:
    dataHeart[i] = LabelEncoder().fit_transform(dataHeart[i])

featureHeart = dataHeart.drop('HeartDisease', axis=1)
labelHeart = dataHeart['HeartDisease']

labelHeart = np.array(labelHeart)
featureHeart = np.array(featureHeart)

li = [0, 3, 4, 7, 9]
for i in li:
    # StandardScaler
    mean = featureHeart[:, i].mean()
    std = featureHeart[:, i].std()
    for num, n in enumerate(featureHeart[:, i]):
        n = (n - mean) / std
        featureHeart[num, i] = n

from imblearn.combine import SMOTEENN

os = SMOTEENN(random_state=42)
featureHeart, labelHeart = os.fit_resample(featureHeart, labelHeart)


def sigmoid(Z):
    A = 1 / (1 + cp.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    A = cp.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = cp.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + cp.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def initialize_parameters(n_x, n_h, n_y):
    cp.random.seed(1)
    W1 = cp.random.randn(n_h, n_x) * 0.01
    b1 = cp.zeros((n_h, 1))
    W2 = cp.random.randn(n_y, n_h) * 0.01
    b2 = cp.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def initialize_parameters_deep(layer_dims):
    cp.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = cp.random.randn(layer_dims[l], layer_dims[l - 1]) / cp.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = cp.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1. / m) * (-cp.dot(Y, cp.log(AL).T) - cp.dot(1 - Y, cp.log(1 - AL).T))
    cost = cp.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * cp.dot(dZ, A_prev.T)
    db = 1. / m * cp.sum(dZ, axis=1, keepdims=True)
    dA_prev = cp.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = - (cp.divide(Y, AL) - cp.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation="sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = cp.zeros((1, m))
    probas, caches = L_model_forward(X, parameters)
    k = 0
    for i in range(0, probas.shape[1]):

        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
        if p[0, i] != y[0, i]:
            # print('i', i)
            # X = cp.array(X)
            # print(X[:, i])
            k += 1
    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    ac = cp.sum((p == y) / m)
    return p, ac, k


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    cp.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)
        dA2 = - (cp.divide(Y, A2) - cp.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 1000 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, cp.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    return parameters, costs


def plot_costs(costs, learning_rate=0.0075):
    plt.plot(cp.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    # plt.show()


from sklearn.model_selection import train_test_split

trainFeature, testFeature, trainLabel, testLabel = train_test_split(featureHeart, labelHeart, random_state=45,
                                                                    test_size=0.2)  # 3
trainFeature = trainFeature.T
trainFeature = cp.array(trainFeature)
trainLabel = trainLabel.reshape(1, -1)
trainLabel = cp.array(trainLabel)
testFeature = testFeature.T
testFeature = cp.array(testFeature)
testLabel = testLabel.reshape(1, -1)
testLabel = cp.array(testLabel)

n_x = 11
n_h = 100
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075
parameters, costs = two_layer_model(trainFeature, trainLabel, layers_dims=(n_x, n_h, n_y), num_iterations=83000,
                                    print_cost=True)

p, ac, k = predict(trainFeature, trainLabel, parameters)
print("Accuracy for TRAIN: ", ac, ' số điểm dữ liệu sai là : ', k)
p, ac, k = predict(testFeature, testLabel, parameters)
print("Accuracy for Test: ", ac, ' số điểm dữ liệu sai là : ', k)

costs = np.array(costs)
plot_costs(costs)