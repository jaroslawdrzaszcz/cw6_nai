########################################################################################################################
# Examples of neuron network for statistic of sobriety data.                                                           #
# Created by Jarosław Drząszcz(s16136).                                                                                #
########################################################################################################################
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle

if __name__ == '__main__':
    # Load statistics of sobriety data
    input_file = "statistics_of_sobriety.txt"
    data = np.loadtxt(input_file, delimiter=',')

    # Shuffle the data
    X, y = shuffle(data[:, :-1], data[:, -1], random_state=7)

    # Split the data into training and testing data
    num_training = int(len(X) * 0.9)
    num_test = len(X) - num_training

    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # First example of neuron network for statistics of sobriety data
    mlp_no1 = MLPRegressor(hidden_layer_sizes=(50,),
                           max_iter=50,
                           alpha=1e-4,
                           solver='sgd',
                           verbose=False,
                           random_state=1,
                           learning_rate_init=.001)
    mlp_no1.fit(X_train, y_train)

    # Second example of neuron network for statistics of sobriety data
    mlp_no2 = MLPRegressor(hidden_layer_sizes=(20,),
                           max_iter=10,
                           alpha=1e-4,
                           verbose=10,
                           random_state=7,
                           learning_rate_init=.1)
    mlp_no2.fit(X_train, y_train)

    # Third example  of neuron network for statistics of sobriety data
    mlp_no3 = MLPRegressor(hidden_layer_sizes=(2,),
                           max_iter=20,
                           alpha=1e-4,
                           solver='lbfgs',
                           verbose=True,
                           random_state=5,
                           learning_rate_init=.01)
    mlp_no3.fit(X_train, y_train)

    print("Training set score for first example of neuron network: %f" % mlp_no1.score(X_train, y_train))
    print("Test set score for first example of neuron network: %f" % mlp_no1.score(X_test, y_test))
    print("Training set score for second example of neuron network: %f" % mlp_no2.score(X_train, y_train))
    print("Test set score for second example of neuron network: %f" % mlp_no2.score(X_test, y_test))
    print("Training set score for third example of neuron network: %f" % mlp_no3.score(X_train, y_train))
    print("Test set score for third example of neuron network: %f" % mlp_no3.score(X_test, y_test))
    test_data = [[80, 1, 35, 300, 2]]
    print("Example of prediction for first neuron network:", mlp_no1.predict(test_data))
    print("Example of prediction for second neuron network:", mlp_no2.predict(test_data))
    print("Example of prediction for third neuron network:", mlp_no3.predict(test_data))
