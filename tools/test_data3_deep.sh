#!/bin/sh

rm -f commviol-test.ds

# creates training dataset for nntool

./dstool -create commviol-test.ds
./dstool -create:141:input commviol-test.ds
./dstool -create:4:output commviol-test.ds
./dstool -import:0 commviol-test.ds commviol.in
./dstool -import:1 commviol-test.ds commviol.out
./dstool -padd:0:meanvar commviol-test.ds
./dstool -padd:1:meanvar commviol-test.ds

##./dstool -list commviol-test.ds

# uses nntool trying to learn from dataset (2 layers only)

# training error is 0.050528/0.011249/0.011249 (deep) [141-500-4]
# ./nntool --deep --samples 100 -v commviol-test.ds 141-500-4 commviol-nn.cfg lbfgs
# ./nntool --deep --samples 100 -v commviol-test.ds 141-100000-4 commviol-nn.cfg lbfgs
## ./nntool --deep=gaussian --samples 100 -v commviol-test.ds 141-1000-4 commviol-nn.cfg grad

# 9 layers deep
ARCH="141-141-141-141-141-141-141-141-141-4"

# overfitting solution
./nntool --time 600 -v commviol-test.ds $ARCH commviol-nn.cfg grad

# training error is 0.042815/0.00981616/0.00981616 (non-deep) [141-500-4]
# training error is 0.041531/0.00981616/3.86269    (non-deep) [141-1000-4]
# training error is 0.055584/0.00981616/2.30756    (non-deep) [141-10000-4]
# ./nntool --samples 100 -v commviol-test.ds 141-1000-4 commviol-nn.cfg lbfgs




# ./nntool --samples 100 --negfb -v commviol-test.ds 141-141-141-141-141-141-141-4 commviol-nn.cfg lbfgs
# ./nntool --samples 100 --load -v commviol-test.ds 141-141-141-141-141-141-141-4 commviol-nn.cfg lbfgs


##################################################
# testing

./nntool -v commviol-test.ds $ARCH  commviol-nn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f commviol-test.ds commviol-pred.ds
./dstool -clear:1 commviol-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v commviol-pred.ds $ARCH commviol-nn.cfg use

./dstool -print:1:2204:2214 commviol-pred.ds
tail commviol.out

echo "ERRORS ARE HIGH HERE AND WE DO *NOT* CONVERGE"


