#!/bin/sh

#### feedforward net result 0.0128355

rm -f commviol-test.ds

# creates training dataset for nntool

./dstool -create commviol-test.ds
./dstool -create:141:input commviol-test.ds
./dstool -create:4:output commviol-test.ds
./dstool -import:0 commviol-test.ds commviol.in
./dstool -import:1 commviol-test.ds commviol.out
./dstool -padd:0:meanvar commviol-test.ds
./dstool -padd:1:meanvar commviol-test.ds

# uses nntool trying to learn from dataset

ARCH="141-141-141-141-4"

### This will result in the KILL (probably out of memory kill).
### Happens before saving neural network. Figure out where it happens exactly
###ARCH="141-500-500-500-500-500-500-500-500-500-500-500-500-500-500-500-500-500-500-500-4"

# DO NOT WORK CURRENTLY (SOME PROBLEMS): ./nntool -v --samples 1000 commviol-test.ds 141-1000-4 commviol-nn.cfg grad

# ./nntool -v --negfb --overfit --samples 1000 --threads 4 commviol-test.ds 141-141-4 commviol-nn.cfg grad

# ./nntool -v --negfb --overfit --time 400 --samples 1000 --threads 4 commviol-test.ds 141-141-4 commviol-nn.cfg random

# ./nntool -v  --load --samples 1000 commviol-test.ds 141-20-20-4 commviol-nn.cfg grad


# ./nntool -v --samples 1000 commviol-test.ds 141-141-4 commviol-nn.cfg mix

# ./nntool -v --samples 1000 commviol-test.ds $ARCH commviol-nn.cfg lbfgs

./nntool -v --time 500 commviol-test.ds $ARCH commviol-nn.cfg grad
# ./nntool -v --time 1000 --threads 1 commviol-test.ds $ARCH commviol-nn.cfg pgrad


#./nntool -v --load  --negfb --samples 1000 commviol-test.ds 141-141-141-141-4 commviol-nn.cfg grad
#./nntool -v --load  --samples 1000 commviol-test.ds 141-141-141-141-4 commviol-nn.cfg lbfgs

# ./nntool -v --samples 500 --load  commviol-test.ds 141-141-4-4 commviol-nn.cfg lbfgs

# ./nntool --samples 10000 --negfb -v commviol-test.ds 141-20-20-20-20-20-20-4 commviol-nn.cfg grad
# ./nntool --samples 10000 --load -v commviol-test.ds 141-20-20-20-20-20-20-4 commviol-nn.cfg grad

##################################################
# testing

./nntool -v commviol-test.ds $ARCH commviol-nn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f commviol-test.ds commviol-pred.ds
./dstool -clear:1 commviol-pred.ds
# ./dstool -remove:1 commviol-pred.ds

./nntool -v commviol-pred.ds $ARCH commviol-nn.cfg use

./dstool -print:1:2204:2214 commviol-pred.ds
tail commviol.out

