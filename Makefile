.PHONY: all clean

export CXX = g++
export CXXFLAGS = -fPIC -Wall -std=c++17

all:
	$(MAKE) -C Fluid

clean:
	$(MAKE) -C Fluid clean