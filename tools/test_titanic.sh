#!/bin/sh
#
# TITANIC training dataset
#

DSTOOL="./dstool"
NNTOOL="./nntool"
DSFILE="titanic-train.ds"
DSFILE_PRED="titanic-test.ds"
DATAINPUT="titanic_train_input.out"
DATAOUTPUT="titanic_train_output.out"
DATAPRED="titanic_test_input.out"
DATAPREDOUT="titanic_test_output.out"
NNFILE="titanic-nn.cfg"
CSVPRED="titanic_pred.csv"


# create empty datafile
$DSTOOL -create $DSFILE
$DSTOOL -create:10:input $DSFILE
$DSTOOL -create:1:output $DSFILE

#populate it with data preprocess using mean=0,var=1 mean/variance removal
$DSTOOL -import:0 $DSFILE $DATAINPUT
$DSTOOL -import:1 $DSFILE $DATAOUTPUT
$DSTOOL -padd:0:meanvar $DSFILE
$DSTOOL -padd:1:meanvar $DSFILE

#######################################################################
# use nntool attempting to learn the dataset using L-BFGS/GRAD learning
ARCH="10-30-30-1"
## ARCH="10-30-30-30-30-30-30-1"


# error was: 0.069748 (using sigmoid non-linearity which gives bad results): 10-50-1 (grad: 50%)
# modified nntool to use tanh(x) and 10-100-100-1 non-linearity: 0.05529/0.066 (grad: 50%)
# modifier nntool to use tanh(x) and 10-500-1 two-layer non-linearity: ~0.068 (grad: 50%)
# nntool leaky ReLu(x) and 10-100-100-1 non-linearity ~0.057882 (grad: 50%)
# nntool leaky ReLu(x) and 10-500-500-500-1 non-linearity and (grad: 75%): 0.058601/0.05879 (77% correct in Kaggle)
# nntool leaky ReLu(x) and 10-500-500-500-1 non-linearity and (lfbgs: 75%): 0.067364/0.629263
# 10-20-20-20-20-1 (grad: 75%): 0.047751/0.0679106
# 10-30-30-30-30-30-30-1 (grad: 100%): 0.031581/0.0395085 [current was: 0.039512]
# nntool 10-1 linear activation function as a test:
##$NNTOOL -v --overfit --samples 100000 $DSFILE $ARCH $NNFILE grad
##$NNTOOL -v --samples 10000 $DSFILE $ARCH $NNFILE grad
##$NNTOOL -v --samples 10000 $DSFILE $ARCH $NNFILE grad

$NNTOOL -v --time 200 $DSFILE $ARCH $NNFILE grad

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

./titanic_kaggle_export.py
