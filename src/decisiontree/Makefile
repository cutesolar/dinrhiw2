# experimental decision tree implemenention (non-elegant, non-optimal)
#
# Tomas Ukkonen
#

CFLAGS=-I. -I.. -g -fsanitize=address
CXXFLAGS=$(CFLAGS)

OBJECTS = decisiontree.o
TST_OBJ = tst/main.o
TARGET  = dtest

######################################################################

all: $(OBJECTS) $(TST_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TST_OBJ) $(OBJECTS)

clean:
	$(RM) $(OBJECTS) $(TST_OBJ) $(TARGET)
	$(RM) *~
