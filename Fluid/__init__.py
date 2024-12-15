from Fluid.SPH.SPH_Solver import SPH_Solver
from Fluid.WCSPH.WCSPH_Solver import WCSPH_Solver
from Fluid.PCISPH.PCISPH_Solver import PCISPH_Solver

from Fluid._basic import *

import taichi as ti

ti.init()

def entry(args):
    solver = None
    if args.method == "sph":
        solver = SPH_Solver()
    elif args.method == "wcsph":
        solver = WCSPH_Solver()
    elif args.method == "pcisph":
        solver = PCISPH_Solver()
    else:
        log(f"{args.method} is not a available algorithm")

    if solver is None:
        return
    
    solver.cmd_args = args

    # build scene with scene json file
    if not solver.build_scene():
        log("cant build scene, exiting...")
        return
    
    solver.run()