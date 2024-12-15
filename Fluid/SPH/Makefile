.PHONY: all clean

all: SPH.o

SPH.o: SPH.cpp SPH.h
	$(CXX) $(CXXFLAGS) -c SPH.cpp -o SPH.o

clean:
	rm -rf *.o