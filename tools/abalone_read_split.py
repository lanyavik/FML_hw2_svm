#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
	xrange = range

os.chdir('C://Users//lanya/libsvm-3.24')

print('data directory', os.getcwd())


def main(argv=sys.argv):
	# transform the 3 classes of the first feature to one-hot vectors
	sex_dic = {"M": ["0", "0", "1"], "F": ["1", "0", "0"], "I": ["0", "1", "0"]}
	dataset, subset_size, subset_file, rest_file =  \
    "datasets/abalone.data", 3133, "datasets/abalone_train.txt", "datasets/abalone_test.txt"
	dataset = open(dataset,'r')
	subset_file = open(subset_file,'w')
	rest_file = open(rest_file,'w')
	
	i = 0
	line = 'flag'
	while i < 4177:
		line = dataset.readline().split(",")
		# transform the labels to {0, 1} binary values
		if 1 <= int(line[8]) <=9:
			line[8] = "1"
		else:
			line[8] = "0"
		line = sex_dic[line[0]] + line[1:]
		if i < subset_size:
			subset_file.write(','.join(line)+'\n')
		else:
			rest_file.write(','.join(line)+'\n')
		i += 1
		print(i)
	subset_file.close()
	rest_file.close()
	dataset.close()

    

if __name__ == '__main__':
	main(sys.argv)

