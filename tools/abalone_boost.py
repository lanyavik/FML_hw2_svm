#!/usr/bin/env python

import os
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from scipy import sparse

from logitboost import LogitBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

os.chdir('C://Users//lanya/libsvm-3.24')
print('data directory', os.getcwd())

def main(plot=True, M=8, n_fold=10):

	"""

	:param plot: whether to plot the train and test result
	:param M: maximum T to search would be 100*M
	:param n_fold: number of folds in cross validation
	:return: None
	"""

	train_file, test_file = "datasets/abalone_train_scaled.txt", "datasets/abalone_test_scaled.txt"

	train = sparse.csr_matrix(np.loadtxt(train_file, delimiter=","))
	test = sparse.csr_matrix(np.loadtxt(test_file, delimiter=","))

	m = train.shape[0]  #3133
	x_dim = train.shape[1] - 1  #10
	x_train, y_train = train[:, :x_dim].toarray(), train[:, x_dim].toarray().squeeze()
	x_test, y_test = test[:, :x_dim].toarray(), test[:, x_dim].toarray().squeeze()
	#print(x_train.shape, y_train.shape, x_test.shape)
	#print(x_train)


	aboost_train_cverror = list(np.ones(M))
	lboost_train_cverror = list(np.ones(M))
	for multiple in range(1, M+1):
		T = multiple * 100
		print("\nT = %s\t" % T)
		# Set AdaBoost parameters
		# decision stump is the default base estimator
		aboost = AdaBoostClassifier(n_estimators=T, random_state=0)
		# Set LogitBoost parameters
		lboost = LogitBoost(n_estimators=T, random_state=0)
		# get 10-fold cross validation error
		aboost_cv_results = cross_validate(aboost, x_train, y_train, cv=n_fold)
		lboost_cv_results = cross_validate(lboost, x_train, y_train, cv=n_fold)
		# compute error by 1 - accuracy
		aboost_train_cverror[multiple-1] = 1 - aboost_cv_results['test_score']
		lboost_train_cverror[multiple-1] = 1 - lboost_cv_results['test_score']
	aboost_train_cverror = np.stack(aboost_train_cverror)
	lboost_train_cverror = np.stack(lboost_train_cverror)
	print(aboost_train_cverror)
	print(lboost_train_cverror)

	# find the T that gives least error (the best cross-validation accuracy)
	a_train_cverror_mean, a_train_cverror_std = aboost_train_cverror.mean(axis=1), aboost_train_cverror.std(axis=1)
	argmin = a_train_cverror_mean.flatten().argmin()
	best_T_aboost = int(argmin+1) * 100
	print("----------------------\n",\
		  "AdaBoost iteration number T = %s\n"%(best_T_aboost), \
		  "----------------------\n")
	# find the T that gives least error (the best cross-validation accuracy)
	l_train_cverror_mean, l_train_cverror_std = lboost_train_cverror.mean(axis=1), lboost_train_cverror.std(axis=1)
	argmin = l_train_cverror_mean.flatten().argmin()
	best_T_lboost = int(argmin + 1) * 100
	print("----------------------\n", \
		  "LogitBoost iteration number T = %s\n" % (best_T_lboost), \
		  "----------------------\n")

	print('Now train with the best T=T* and eval on the test set\n')

	# Train on the whole train set
	aboost = AdaBoostClassifier(n_estimators=best_T_aboost, random_state=0)
	aboost.fit(x_train, y_train)
	lboost = LogitBoost(n_estimators=best_T_lboost, random_state=0)
	lboost.fit(x_train, y_train)
	# Test on the test set
	y_pred_train = aboost.predict(x_train)
	y_pred_test = aboost.predict(x_test)
	a_error_train = 1-accuracy_score(y_train, y_pred_train)
	a_error_test = 1-accuracy_score(y_test, y_pred_test)
	print("AdaBoost train error: %s test error: %s" % (a_error_train, a_error_test))

	y_pred_train = lboost.predict(x_train)
	y_pred_test = lboost.predict(x_test)
	l_error_train = 1-accuracy_score(y_train, y_pred_train)
	l_error_test = 1-accuracy_score(y_test, y_pred_test)
	print("LogitBoost train error: %s test error: %s"%(l_error_train, l_error_test))


	if plot:
		plt.figure()
		x_values = range(100, M*100+1, 100)
		plt.plot(x_values, a_train_cverror_mean, label="AdaBoost")
		plt.fill_between(x_values,
						a_train_cverror_mean + a_train_cverror_std,
						a_train_cverror_mean - a_train_cverror_std,
						alpha=0.5, edgecolor='blue', facecolor='blue')
		plt.plot(x_values, l_train_cverror_mean, label="LogitBoost")
		plt.fill_between(x_values,
						l_train_cverror_mean + l_train_cverror_std,
						l_train_cverror_mean - l_train_cverror_std,
						alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848')
		plt.xlabel('T (number of iterations/classifiers)')
		plt.ylabel('10fold cross validation train error')
		plt.legend()
		plt.ylim(0, 0.5)
		plt.savefig('B.i_cverror.png')



    
if __name__ == '__main__':
	main()

