.PHONY: all SPH WCSPH PCISPH clean

all: SPH

SPH:
	$(MAKE) -C SPH
	$(CXX) -shared -o _importer/libsph.so SPH/*.o

clean:
	$(MAKE) -C SPH clean
	rm -rf _importer/*.so