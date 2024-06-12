#!/bin/sh

autoheader
autoconf
# in configure make sure that proper BLAS library is found and selected (use configure --help)..
./configure
make depend
make -j3 make_objects
make makelib

# then do make install as a root to INSTALL








