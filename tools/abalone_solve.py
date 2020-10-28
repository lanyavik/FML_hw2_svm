#!/usr/bin/env python

import os
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from scipy import sparse

os.chdir('C://Users//lanya/libsvm-3.24')
print('data directory', os.getcwd())
from python.svmutil import svm_train, svm_predict

def main(plot=True, log2c_range=15, n_fold=10, poly_d=4, task="0"):

	"""

	:param plot: whether to plot the train and test result
	:param log2c_range: k s.t. C=2^k
	:param n_fold: number of folds in cross validation
	:param poly_d: degree of polynomial kernel
	:param task: "1" - solves C.4 & C.5
				 "0" - solves C.6.d
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


	train_cverror_lst = np.ones((poly_d, 2 * log2c_range + 1, n_fold))
	for d in range(1, poly_d+1):
		if task == "0":
			# Apply the formula (*) in FML-FA20-HW2-xc1305-xinyue-chen.pdf
			x_train_matrix = np.power(np.dot(x_train, (np.repeat(y_train.reshape(-1, 1), x_dim, axis=1) * x_train).transpose()), d)
			x_test_matrix = np.power(np.dot(x_test, (np.repeat(y_train.reshape(-1, 1), x_dim, axis=1) * x_train).transpose()), d)
			x_train_matrix, x_test_matrix = sparse.csr_matrix(x_train_matrix), sparse.csr_matrix(x_test_matrix)
		else:
			x_train_matrix, x_test_matrix = x_train, x_test
		for log2c in range(-log2c_range, log2c_range+1):
			#if task == "0" and log2c < 10:
			#	continue
			print("\n(C,d) = (2^%s, %s)\t" % (log2c, d))
			# Do Cross Validation, first shuffle the list
			indexList = np.arange(m)
			np.random.shuffle(indexList)
			x_train_shuffle, y_train_shuffle = x_train_matrix[indexList, :], y_train[indexList]
			for f in range(n_fold):
				i1, i2 = m * f // n_fold, m * (f + 1) // n_fold
				# eval using the f-th
				x_eval, y_eval = x_train_shuffle[i1:i2, :], y_train_shuffle[i1:i2]
				# train using the rest f-1 folds
				x_trn = sparse.vstack([x_train_shuffle[:i1, :], x_train_shuffle[i2:, :]]).toarray()
				y_trn = np.hstack([y_train_shuffle[:i1], y_train_shuffle[i2:]])
				#print(x_train_shuffle[:i1, :].shape, x_train_shuffle[i2:, :].shape, x_trn.shape, )
				#print(x_trn)
				# Train with default C-SVM
				model = svm_train(y_trn, x_trn, "-c %s -t %s -d %s -q" % (2**log2c, task, d))
				eval_predy, eval_stats, eval_predval = svm_predict(y_eval, x_eval, model)
				# libsvm svm_predict() returns accuracy percnetage as eval_stats[0],
				# compute error by 1-p/100
				train_cverror_lst[d-1][log2c+log2c_range][f] = 1 - eval_stats[0]/100  # record error

	# find the pair (C,d) that gives least error (the best cross-validation accuracy)
	train_cverror_mean, train_cverror_std = train_cverror_lst.mean(axis=2), train_cverror_lst.std(axis=2)
	argmin = train_cverror_mean.flatten().argmin()
	best_log2c, best_d = int(argmin)%(2*log2c_range+1)-log2c_range, int(argmin)//(2*log2c_range+1)+1
	best_C = 2 ** best_log2c
	print("----------------------\n",\
		  "Best (C,d) = (%s, %s)\n"%(best_C, best_d), \
		  "----------------------\n")
	bestc_train_cverror = train_cverror_mean[:, best_log2c+log2c_range]

	print('Now train with the best C=C* and eval on the test set\n')
	bestc_test_error = np.zeros([4, 1])
	sv_num = np.zeros([4, 1])
	for d in range(1, poly_d + 1):
		if task == "0":
			# Apply the formula (*) in FML-FA20-HW2-xc1305-xinyue-chen.pdf
			x_train_matrix = np.power(np.dot(x_train, (np.repeat(y_train.reshape(-1, 1), x_dim, axis=1) * x_train).transpose()), d)
			x_test_matrix = np.power(np.dot(x_test, (np.repeat(y_train.reshape(-1, 1), x_dim, axis=1) * x_train).transpose()), d)
			x_train_matrix, x_test_matrix = sparse.csr_matrix(x_train_matrix), sparse.csr_matrix(x_test_matrix)
		else:
			x_train_matrix, x_test_matrix = x_train, x_test

		# Train on the whole train set
		model = svm_train(y_train, x_train_matrix, " -c %s -t %s -d %s " % (best_C, task, d))
		# Test on the test set
		test_predy, test_stats, test_predval = svm_predict(y_test, x_test_matrix, model)
		bestc_test_error[d-1] = 1 - test_stats[0]/100
		sv_num[d-1] = model.get_nr_sv()

	print(train_cverror_mean)
	print(train_cverror_std)
	print(bestc_train_cverror)
	print(bestc_test_error)
	print(sv_num)

	if plot and task == '1':
		for d in range(1, poly_d+1):
			plt.figure()
			plt.plot(range(-log2c_range, log2c_range+1), train_cverror_mean[d-1])
			plt.fill_between(range(-log2c_range, log2c_range+1),
							train_cverror_mean[d-1] + train_cverror_std[d-1],
							train_cverror_mean[d-1] - train_cverror_std[d-1],
							alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
			plt.xlabel('log2C (Cost) ')
			plt.ylabel('10fold cross validation train error')
			plt.savefig('C.4_cverror_d=%s.png'%d)

		plt.figure()
		plt.plot(range(1, 5), bestc_train_cverror)
		plt.xlabel('d (degree of kernel)')
		plt.ylabel('10fold cross validation train error')
		plt.savefig('C.5_trainerror.png')

		plt.figure()
		plt.plot(range(1, 5), bestc_test_error)
		plt.xlabel('d (degree of kernel)')
		plt.ylabel('test error')
		plt.savefig('C.5_testerror.png')

		plt.figure()
		plt.plot(range(1, 5), sv_num)
		plt.xlabel('d (degree of kernel)')
		plt.ylabel('num support vectors')
		plt.savefig('C.5_numsv.png')

		#bsv_num = (input("bsv> ").split())
		#msv_num = sv_num - np.array(bsv_num)
		#print('msv: ', msv_num)

	if plot and task == '0':
		plt.figure()
		plt.plot(range(1, 5), bestc_train_cverror)
		plt.xlabel('d (degree of kernel)')
		plt.ylabel('10fold cross validation train error')
		plt.savefig('C.6d_trainerror.png')

		plt.figure()
		plt.plot(range(1, 5), bestc_test_error)
		plt.xlabel('d (degree of kernel)')
		plt.ylabel('test error')
		plt.savefig('C.6d_testerror.png')


    
if __name__ == '__main__':
	main()

