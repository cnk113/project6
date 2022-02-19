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

	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.7)
	reg = logreg.LogisticRegression(
        X_train.shape[1], max_iter=1000, learning_rate=0.001, batch_size=12, tol=1e-6
    )
	reg.train_model(X_train, y_train, X_val, y_val)
	loss = reg.loss_history_train
	assert loss[-1] < loss[0], "Loss didn't decrease"

	reg = logreg.LogisticRegression(
        X_train.shape[1], max_iter=2, learning_rate=0.001, batch_size=12, tol=1e-6
    )
	losses = 0
	for i in range(5):
		reg.train_model(X_train, y_train, X_val, y_val)
		losses += reg.loss_history_train[-1]
	assert loss[-1] <= losses/5, "Loss should be lower after longer training"


def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	X_train, X_val, y_train, y_val = utils.loadDataset(split_percent=0.7)
	reg = logreg.LogisticRegression(
		X_train.shape[1], max_iter=1000, learning_rate=0.001, batch_size=12, tol=1e-6
    )
	w = reg.W
	reg.train_model(X_train, y_train, X_val, y_val)
	w2 = reg.W
	assert not np.allclose(w, w2), "Weights didn't change after training"

	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	pred = reg.make_prediction(X_val)
	pred[pred >= .5] = 1
	pred[pred < .5] = 0
	acc = np.sum(y_val == pred) / len(pred)
	assert acc > .5, "Accuracy is worse than guessing"
