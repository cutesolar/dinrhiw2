#!/bin/sh
#
# MNIST training dataset (42.000 test cases, 4200 per each digit)
#

DSTOOL="./dstool"
NNTOOL="./nntool"
DSFILE="mnist-train.ds"
DSFILE_PRED="mnist-test.ds"
DATAINPUT="mnist_digit_train_input.out"
DATAOUTPUT="mnist_digit_train_output.out"
DATAPRED="mnist_digit_test_input.out"
DATAPREDOUT="mnist_digit_test_output.out"
NNFILE="mnist-nn.cfg"
CSVPRED="mnist_digit_pred.csv"

rm -f $DSFILE

## create empty datafile
$DSTOOL -create $DSFILE
$DSTOOL -create:784:input $DSFILE
$DSTOOL -create:10:output $DSFILE

##populate it with data preprocess using mean=0,var=1 mean/variance removal
$DSTOOL -import:0 $DSFILE $DATAINPUT
$DSTOOL -import:1 $DSFILE $DATAOUTPUT
$DSTOOL -padd:0:meanvar $DSFILE
$DSTOOL -padd:1:meanvar $DSFILE

#######################################################################
# use nntool attempting to learn the dataset using L-BFGS/GRAD learning
ARCH="784-300-300-300-300-10"

# 784-300-300-300-300-10 (grad: 75%): 
## $NNTOOL -v --overfit --samples 10000 $DSFILE $ARCH $NNFILE grad
$NNTOOL -v --overfit --time 3600 --threads 1 $DSFILE $ARCH $NNFILE grad
## $NNTOOL -v --samples 10000 $DSFILE $ARCH $NNFILE grad

# Try bayesian neural network sampling from overfitten best solution.
# This will take into account uncertainty. DO NOT WORK WELL
#### $NNTOOL -v --samples 10000 --adaptive --load $DSFILE $ARCH $NNFILE bayes

#######################################################################
# test NN performance

$NNTOOL -v $DSFILE $ARCH $NNFILE use

#######################################################################
# predict scores

cp -f $DSFILE $DSFILE_PRED
$DSTOOL -clear:0 $DSFILE_PRED
$DSTOOL -clear:1 $DSFILE_PRED

# import new data to DSFILE_PRED
$DSTOOL -import:0 $DSFILE_PRED $DATAPRED

# calculates predictions
$NNTOOL -v $DSFILE_PRED $ARCH $NNFILE use

$DSTOOL -export:1 $DSFILE_PRED $DATAPREDOUT

#######################################################################
# converts $DATAPREDOUT TO KAGGLE SUBMISSION CSVFILE

./mnist_kaggle_export.py

