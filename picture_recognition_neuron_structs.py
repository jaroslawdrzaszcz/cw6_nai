########################################################################################################################
# Examples of neuron network for digit recognition from pictures.                                                      #
# Created by PraemtsJarosław Drząszcz(s16136) and Przemysław Białczak(s16121)                                          #
########################################################################################################################

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

if __name__ == '__main__':
    # Load digits data
    data = datasets.load_digits()

    # Shuffle the data
    X, y = shuffle(data.data, data.target, random_state=7)

    # Split the data into training and testing datasets
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # First example of neuron network for digits data
    mlp_no1 = MLPClassifier(hidden_layer_sizes=(50,),
                            max_iter=50,
                            alpha=1e-4,
                            solver='sgd',
                            verbose=False,
                            random_state=1,
                            learning_rate_init=.001)
    mlp_no1.fit(X_train, y_train)

    # Second example of neuron network for digits data
    mlp_no2 = MLPClassifier(hidden_layer_sizes=(20,),
                            max_iter=10,
                            alpha=1e-4,
                            verbose=10,
                            random_state=7,
                            learning_rate_init=.1)
    mlp_no2.fit(X_train, y_train)

    # Third example  of neuron network for digits data
    mlp_no3 = MLPClassifier(hidden_layer_sizes=(2,),
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
    test_data = [[0.,  0.,  5., 4., 0.,  0.,  0.,  0., 0.,  0.,  0., 12., 10., 10.,  0.,  0., 0.,  0.,  0., 10., 16.,
                  16.,  9.,  0.,0.,  0.,  0., 6., 14., 12.,  6.,  0., 0.,  0.,  1., 4., 10.,  10.,  2.,  0., 0.,  0.,
                  1., 1., 5.,  6.,  0.,  0., 0.,  0.,  2., 5., 8., 12.,  0.,  0., 0.,  0., 10., 11., 12., 12.,  1.,  0.]]
    print("Example of prediction for first neuron network:", mlp_no1.predict(test_data))
    print("Example of prediction for second neuron network:", mlp_no2.predict(test_data))
    print("Example of prediction for third neuron network:", mlp_no3.predict(test_data))
