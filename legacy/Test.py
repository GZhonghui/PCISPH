'''

Number of particles: 0
---------------------------------------------------
Begin updating frame: 0 timeIntervalInSeconds: 0.0333333 (1/30) seconds
Update collider took 6e-06 seconds
Number of newly generated particles: 201280
Number of total generated particles: 201280
Update emitter took 0.34564 seconds
Using adaptive sub-timesteps
Number of remaining sub-timesteps: 24
Begin onAdvanceTimeStep: 0.00138889 (1/720) seconds
Update collider took 1e-06 seconds
Update emitter took 0 seconds
Average number of points per non-empty bucket: 76.2424
Max number of points per bucket: 128
Building neighbor searcher took: 0.116536 seconds
Building neighbor list took: 1.04451 seconds
Building neighbor lists and updating densities took 1.19486 seconds
Main Start --------
Spacing0.02
R0.036
stepTime0.00138889
mass0.00403914
D1000
Delta 152.934
Main End   --------
Number of PCI iterations: 1
Max density error after PCI iteration: 0.0327587
Accumulating forces took 0.143037 seconds
Time integration took 0.000707 seconds
Resolving collision took 0.110773 seconds
Max density: 1000 Max density / target density ratio: 1
End onAdvanceTimeStep (took 1.4659 seconds)
Number of remaining sub-timesteps: 23
Begin onAdvanceTimeStep: 0.00138889 (1/720) seconds
Update collider took 2e-06 seconds
Update emitter took 0 seconds
Average number of points per non-empty bucket: 76.2424
Max number of points per bucket: 128
Building neighbor searcher took: 0.107061 seconds
Building neighbor list took: 0.295308 seconds
Building neighbor lists and updating densities took 0.437317 seconds
Main Start --------
Spacing0.02
R0.036
stepTime0.00138889
mass0.00403914
D1000
Delta 152.934
Main End   --------
Number of PCI iterations: 1
Max density error after PCI iteration: 0.131271
Accumulating forces took 0.147649 seconds
Time integration took 0.000798 seconds
Resolving collision took 0.110875 seconds
Max density: 1000.02 Max density / target density ratio: 1.00002
End onAdvanceTimeStep (took 0.709242 seconds)
Number of remaining sub-timesteps: 22
Begin onAdvanceTimeStep: 0.00138889 (1/720) seconds
Update collider took 1e-06 seconds
Update emitter took 0 seconds
Average number of points per non-empty bucket: 76.2424
Max number of points per bucket: 128
Building neighbor searcher took: 0.105561 seconds
Building neighbor list took: 0.289136 seconds
Building neighbor lists and updating densities took 0.429658 seconds
Main Start --------
Spacing0.02
R0.036
stepTime0.00138889
mass0.00403914
D1000
Delta 152.934
Main End   --------
Number of PCI iterations: 1
Max density error after PCI iteration: 0.299465
Accumulating forces took 0.149226 seconds
Time integration took 0.00097 seconds
Resolving collision took 0.110926 seconds
Max density: 1000.09 Max density / target density ratio: 1.00009
End onAdvanceTimeStep (took 0.703249 seconds)
Number of remaining sub-timesteps: 21
Begin onAdvanceTimeStep: 0.00138889 (1/720) seconds
Update collider took 1e-06 seconds
Update emitter took 0 seconds
Average number of points per non-empty bucket: 76.2424
Max number of points per bucket: 128
Building neighbor searcher took: 0.099566 seconds

D:\Codes\Git\FluidEngine\x64\Release\FluidExample.exe (process 19784) exited with code -1073741510.
Press any key to close this window . . .

'''

EPS = 1e-10

timeStep = 0.00138889

kernelR = 0.036
kernelR_2 = kernelR * kernelR
kernelR_3 = kernelR_2 * kernelR
kernelR_4 = kernelR_2 * kernelR_2
kernelR_5 = kernelR_2 * kernelR_3

particleMass = 0.00403914
particleMass_2 = particleMass * particleMass

Spacing = 0.02

searchR = Spacing * 1.2

@ti.func
def Square(x):
  return x * x

@ti.func
def Spiky(r):
  Res = 0.0
  r = max(0, r)
  if r <= kernelR:
    A = 15 / tiPi / kernelR_3
    B = (1 - r / kernelR)
    Res = A * B * B * B
  return Res

@ti.func
def SpikyFirstDerivative(r):
  Res = 0.0
  r = max(0, r)
  if r <= kernelR:
    A = -45 / tiPi / kernelR_4
    B = (1 - r / kernelR)
    Res = A * B * B
  return Res

@ti.func
def SpikySecondDerivative(r):
  Res = 0.0
  r = max(0, r)
  if r <= kernelR:
    A = 90 / tiPi / kernelR_5
    B = 1 - r / kernelR
    Res = A * B
  return Res

@ti.func
def SpikyGradient(p):
  Res = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
  Distance = p.norm()
  if Distance > EPS:
    Res = -SpikyFirstDerivative(Distance) * p / Distance
  return Res

def main():
    pass

if __name__=='__main__':
    main()