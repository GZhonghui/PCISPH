# right hand, Y up

from Fluid.SPH.SPH_Solver import SPH_Solver
from Fluid.WCSPH.WCSPH_Solver import WCSPH_Solver
from Fluid.PCISPH.PCISPH_Solver import PCISPH_Solver

from Fluid.Render.Renderer import Renderer

from Fluid._basic import *

import taichi as ti
import mitsuba as mi

# global init
ti.init()
mi.set_variant("scalar_rgb")

def simulation_entry(args):
    log("start simulation...")
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

def render_entry(args):
    log("start rendering...")
    renderer = Renderer(args)
    renderer.render_all()

def build_surface_entry(args):
    log("start building fliud surface...")