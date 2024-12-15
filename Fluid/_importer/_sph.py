__all__ = [
    "_c_sph_init"
]

import ctypes, os

current_path = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(current_path, 'libsph.so')
libsph = ctypes.cdll.LoadLibrary(so_path)

_c_sph_init = libsph.sph_init
_c_sph_init.argtypes = ()
_c_sph_init.restype = None