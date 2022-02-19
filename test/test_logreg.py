"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
from regression import utils, logreg
import numpy as np

def test_updates():
	"""

	"""
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	reg = logreg.LogisticRegression(
        X_train.shape[1], max_iter=1000, learning_rate=0.001, batch_size=32, tol=1e-6
    )
	reg.train_model(X_train, y_train, X_test, y_test)
	loss = reg.loss_history_train
	assert loss[-1] < loss[0], "Loss didn't decrease"

	reg = logreg.LogisticRegression(
        X_train.shape[1], max_iter=10, learning_rate=0.001, batch_size=32, tol=1e-6
    )
	reg.train_model(X_train, y_train, X_test, y_test)
	loss2 = reg.loss_history_train
	assert loss[-1] <= loss2[-1], "Loss should be lower after longer training"


def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	reg = logreg.LogisticRegression(
		X_train.shape[1], max_iter=1000, learning_rate=0.001, batch_size=32, tol=1e-6
    )
	w = reg.W
	reg.train_model(X_train, y_train, X_test, y_test)
	w2 = reg.W
	assert not np.allclose(w, w2), "Weights didn't change after training"

	pred = reg.make_prediction(X_test)
	pred[pred >= .5] = 1
	pred[pred < .5] = 0
	acc = np.sum(y_test == pred) / len(pred)
	assert acc > .5, "Accuracy is worse than guessing"
