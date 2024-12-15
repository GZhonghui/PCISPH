from Fluid.SPH.SPH_Solver import SPH_Solver

class PCISPH_Solver(SPH_Solver):
    def __init__(self):
        super().__init__()

    def __del__(self):
        ...