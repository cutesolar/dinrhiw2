# dinrhiw2 reinforcement learning
##################################################

CC = gcc
CXX= g++

OBJECTS = diffeqs.o HMC_diffeq.o

EXTRA_OBJECTS = 

TEST_OBJECTS = $(OBJECTS) $(EXTRA_OBJECTS) tst/test.o

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -mtune=native -ftree-vectorize -fopenmp -mtune=native -march=native
CFLAGS=-g -O2 -Wno-deprecated -Wno-strict-aliasing -Wno-attributes -Wno-class-memaccess -I/usr/include/ -I/usr/local/include -m64 -DUSE_SDL `pkg-config sdl2 --cflags` `pkg-config dinrhiw --cflags`

# -fsanitize=address

TARGET=testsuite

CXXFLAGS=-Wall $(CFLAGS) -g -I. -I.. -I../.. -Iexceptions -I../math -I../neuralnetwork -I../hmm -I../rnn -I../crypto $(OPTIMIZE)

LIBS=-L/usr/local/lib -lgmp -lpthread -L/usr/lib/  `pkg-config dinrhiw --libs`

LIBS_PYTHON=`python3-config --embed --ldflags` `pkg-config sdl2 --libs`

CD = cd


##################################################

all: $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TEST_OBJECTS) $(LIBS) $(LIBS_PYTHON)

clean:
	$(RM) $(TEST_OBJECTS)
	$(RM) $(TARGET)
	$(RM) *~

realclean: clean

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend


##################################################





