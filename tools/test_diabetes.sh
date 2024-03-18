#!/bin/bash
#
# DIABETES training dataset from Kaggle (public domain dataset)
#

./diabetes_preprocess.py

DSTOOL="./dstool"
NNTOOL="./nntool"
DSFILE="diabetes-train.ds"
DSFILE_PRED="diabetes-test.ds"
DATAINPUT="diabetes_input.csv"
DATAOUTPUT="diabetes_output.csv"
NNFILE="diabetes-nn.cfg"

# create empty datafile
$DSTOOL -create $DSFILE
$DSTOOL -create:8:input $DSFILE
$DSTOOL -create:1:output $DSFILE

#populate it with data preprocess using mean=0,var=1 mean/variance removal'
$DSTOOL -import:0 $DSFILE $DATAINPUT
$DSTOOL -import:1 $DSFILE $DATAOUTPUT
$DSTOOL -padd:0:meanvar $DSFILE
## $DSTOOL -padd:1:meanvar $DSFILE

#######################################################################
# use nntool attempting to learn the dataset using L-BFGS/GRAD learning
## ARCH="8-1"
ARCH="8-30-30-30-30-30-30-30-30-30-1"

$NNTOOL -v --time 200 $DSFILE $ARCH $NNFILE grad
#$NNTOOL -v --time 200 --overfit --noresidual $DSFILE $ARCH $NNFILE pgrad

#######################################################################
# test NN performance

$NNTOOL -v $DSFILE $ARCH $NNFILE use

