# FML_hw2_svm

Code implementation are under the folder "./tools" of libsvm

Order of code execution: 

`python abalone_read_split.py` output> "abalone_train.txt", "abalone_test.txt"

`python abalone_scale.py` output> "abalone_train_scaled.txt", "abalone_test_scale.txt"

`python abalone_solve.py` 
-- `task="1"` solves question C.4 & C.5
   output(optional)> "C.4_cverror_d=1.png"... "C.4_cverror_d=4.png", "C.5_trainerror.png", "C.5_testerror.png", "C.5_numsv.png"
 
-- `task="1"` solves question C.6d
   output(optional)> "C.6d_trainerror.png"... "C.6d_testerror.png"


Preprocessed datasets and the plots are also provided for your reference.
