# Datamining Makefile
###################################################

CC = gcc
CXX= g++

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -mtune=native -ftree-vectorize -fopenmp -mtune=native -march=native
CFLAGS += $(OPTIMIZE) -Wall -g -I. -I.. -I../../ -I../neuralnetwork -I../math -I/usr/include/ -I/usr/local/include/ -DOPENBLAS -I/ucrt64/include/openblas/ 
CXXFLAGS += $(CFLAGS) -I@atlas_headerpath@


OBJECTS = AssociationRuleFinder.o FrequentSetsFinder.o discretize.o test/tst.o

EXTRA_OBJECTS = ../dynamic_bitset.o ../timed_boolean.o ../Log.o ../math/vertex.o ../math/matrix.o ../math/ownexception.o ../math/integer.o \
	../math/matrix_rotations.o ../math/eig.o ../math/correlation.o ../math/blade_math.o \
	../math/real.o 	../dataset.o ../conffile.o ../linear_ETA.o \
	../math/ica.o ../math/BFGS.o ../math/LBFGS.o ../math/linear_algebra.o \
	../math/correlation.o ../math/ica.o ../math/linear_equations.o ../math/norms.o ../math/RNG.o \
	../math/modular.o ../math/superresolution.o \
	../math/outerproduct.o ../math/fastpca.o ../VisualizationInterface.o \
	../math/quaternion.o ../math/SGD.o ../math/RungeKutta.o ../math/pocketfft/pocketfft.o


SOURCES = AssociationRuleFinder.cpp test/tst.cpp ../math/RNG.cpp \
	../dynamic_bitset.cpp ../timed_boolean.cpp ../math/ownexception.cpp ../math/integer.cpp \
	../math/modular.cpp ../math/superresolution.cpp


TARGET  = tst

LIBS= -L/usr/local/lib -lgmp -L/usr/lib/  -L/ucrt64/lib/ -lopenblas -lstdc++ -lpython3.11 -ldl  -lm 

##################################################

all: $(OBJECTS) $(EXTRA_OBJECTS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJECTS) $(EXTRA_OBJECTS) $(LIBS)

clean:
	$(RM) $(OBJECTS) $(TARGET)
	$(RM) *~ test/*~

clean_rpo:
	$(RM) *.rpo
	$(RM) test/*.rpo

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend

##################################################

include Makefile.depend
