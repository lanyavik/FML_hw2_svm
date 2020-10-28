#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict
import numpy as np
from scipy import sparse

os.chdir('C://Users//lanya/libsvm-3.24')

print('data directory', os.getcwd())

from python.commonutil import csr_find_scale_param, csr_scale

def main(argv=sys.argv):

	train_file, test_file = "datasets/abalone_train.txt", "datasets/abalone_test.txt"

	x_train = sparse.csr_matrix(np.loadtxt(train_file, delimiter=","))
	x_test = sparse.csr_matrix(np.loadtxt(test_file, delimiter=","))

	print(x_train.shape, x_test.shape)
	for i in range(x_train.shape[1]-1):
		param = csr_find_scale_param(x_train[:, i], lower=0)
		x_train[:, i] = csr_scale(x_train[:, i], param)
		x_test[:, i] = csr_scale(x_test[:, i], param)

	print(x_train)
	np.savetxt(train_file.replace('.txt', '_scaled.txt'), x_train.toarray(), delimiter=',')
	np.savetxt(test_file.replace('.txt', '_scaled.txt'), x_test.toarray(), delimiter=',')

if __name__ == '__main__':
	main(sys.argv)

