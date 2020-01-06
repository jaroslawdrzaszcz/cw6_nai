########################################################################################################################
# Examples of neuron network for irises data.                                                                          #
# Created by Jarosław Drząszcz(s16136) and Przemysław Białczak(s16121).                                                #
########################################################################################################################

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

if __name__ == '__main__':
    # Load Statistics of irises
    data = datasets.load_iris()

    # Shuffle the data
    X, y = shuffle(data.data, data.target, random_state=7)

    # Split the data into training and testing datasets
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # First example of neuron network for irises data
    mlp_no1 = MLPClassifier(hidden_layer_sizes=(50,),
                            max_iter=50,
                            alpha=1e-4,
                            solver='sgd',
                            verbose=False,
                            random_state=1,
                            learning_rate_init=.001)
    mlp_no1.fit(X_train, y_train)

    # Second example of neuron network for irises data
    mlp_no2 = MLPClassifier(hidden_layer_sizes=(20,),
                            max_iter=10,
                            alpha=1e-4,
                            verbose=10,
                            random_state=7,
                            learning_rate_init=.1)
    mlp_no2.fit(X_train, y_train)

    # Third example  of neuron network for irises data
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

    print("Example of prediction for first neuron network:", mlp_no1.predict([[4, 3, 2, 1]]))
    print("Example of prediction for second neuron network:", mlp_no2.predict([[5.9, 3., 5.1, 1.8]]))
    print("Example of prediction for third neuron network:", mlp_no3.predict([[4.5, 3.2, 4.4, 1.5]]))

    # fig, axes = plt.subplots(4, 4)
    # # use global min / max to ensure all weights are shown on the same scale
    # vmin, vmax= mlp.coefs_[0].min(), mlp.coefs_[0].max()
    # for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    #     ax.matshow(coef.reshape(2, 2), cmap='RdBu_r', vmin=.5 * vmin, vmax=.5 * vmax)
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #
    # plt.show()
