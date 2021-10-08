import taichi as ti

ti.init(arch=ti.cuda)

tiPi = ti.acos(-1.0)

Output = True

timeStep = 0.001

maxPC = 10
allowRoError = 0.1

FPS = 30

Ro = 1.4147106

boxX = (-40.0, 40.0)
boxY = (-40.0, 40.0)
boxZ = ( 0.0,  30.0)

particleX = 120
particleY = 120
particleZ = 80

particleCnt = particleX * particleY * particleZ

particleMass = 0.001
particleMass_2 = particleMass * particleMass

viscosityCoefficient = 0

kernelR = 0.15
kernelR_2 = kernelR * kernelR
kernelR_3 = kernelR_2 * kernelR
kernelR_4 = kernelR_2 * kernelR_2
kernelR_5 = kernelR_2 * kernelR_3

searchR = kernelR * 1.2

gridX = (boxX[1] - boxX[0]) // (searchR * 2) + 1
gridY = (boxY[1] - boxY[0]) // (searchR * 2) + 1
gridZ = (boxZ[1] - boxZ[0]) // (searchR * 2) + 1

gridCnt = int(gridX * gridY * gridZ)

maxParticlesPerGrid = 16

maxRo = ti.field(dtype=ti.f32, shape=())

Gravity = ti.Vector([0.0, 0.0, -9.8], dt=ti.f32)

particleLocations  = ti.Vector.field(3, dtype=ti.f32, shape=(particleCnt,2))
particleDensities  = ti.field(dtype=ti.f32, shape=(particleCnt,2))
particleExForces   = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particleVelocities = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particlePressures  = ti.field(dtype=ti.f32, shape=particleCnt)
particlePressureFs = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particleSumForces  = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)

gridParticles = ti.field(dtype=ti.i32, shape=(gridCnt, maxParticlesPerGrid))

@ti.func
def gridIndex(Location):
  xIndex = ti.cast((Location[0] - boxX[0]) // (searchR * 2), ti.i32)
  yIndex = ti.cast((Location[1] - boxY[0]) // (searchR * 2), ti.i32)
  zIndex = ti.cast((Location[2] - boxZ[0]) // (searchR * 2), ti.i32)
  return ti.Vector([xIndex, yIndex, zIndex], dt=ti.i32)

@ti.func
def gridIndexInOne(Indexs):
  Res = ti.cast(Indexs[0] * gridY * gridZ + Indexs[1] * gridZ + Indexs[2], ti.i32)
  if Indexs[0] < 0 or Indexs[1] < 0 or Indexs[2] < 0:
    Res = -1
  if Indexs[0] >= gridX or Indexs[1] >= gridY or Indexs[2] >= gridZ:
    Res = -1
  return Res

@ti.func
def nextGrid(Index, i):
  Index[0] += (i %  3) - 1
  Index[1] += (i // 3) % 3 -1
  Index[2] += (i // 9) - 1
  return Index

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
  if Distance > 0.0:
    Res = -SpikyFirstDerivative(Distance) * p / Distance
  return Res

@ti.func
def computeBeta():
  return 2 * Square(particleMass * timeStep / Ro)

@ti.func
def computeDelta():
  return 1

@ti.kernel
def InitTaichi():
  for i in range(particleCnt):
    xIndex = (i %  particleX)
    yIndex = (i // particleX) % particleY
    zIndex = (i // (particleX * particleY))
    particleLocations[i][0] = -10 + kernelR * xIndex
    particleLocations[i][1] = -10 + kernelR * yIndex
    particleLocations[i][2] =  15 + kernelR * zIndex
    particleVelocities[i] = (0, 0, 0)    

@ti.kernel
def computeDensities(Pred):
  maxRo[None] = 0
  for i in range(particleCnt):
    Sum = 0.0
    thisIndex = gridIndex(particleLocations[i,Pred])
    for k in range(27):
      newIndex = nextGrid(thisIndex, k)
      Idx = gridIndexInOne(newIndex)
      if not Idx == -1:
        for j in range(1,maxParticlesPerGrid):
          if j > gridParticles[Idx,0]:
            break
          Target = gridParticles[Idx,j]
          D = (particleLocations[i,Pred] - particleLocations[Target,Pred]).norm()
          Sum += Spiky(D)
    particleDensities[i,Pred] = particleMass * Sum
    maxRo[None] = max(maxRo[None], particlePredictD[i,Pred])

@ti.kernel
def computeGravityForces():
  for i in range(particleCnt):
    particleExForces[i]  = Gravity * particleMass
    particleSumForces[i] = particleExForces[i]

@ti.kernel
def computeViscosityForces():
  for i in range(particleCnt):
    thisIndex = gridIndex(particleLocations[i,0])
    for k in range(27):
      newIndex = nextGrid(thisIndex, k)
      Idx = gridIndexInOne(newIndex)
      if not Idx == -1:
        for j in range(1,maxParticlesPerGrid):
          if j > gridParticles[Idx,0]:
            break
          Target = gridParticles[Idx,j]
          Distance = (particleLocations[i] - particleLocations[Target]).norm()
          Force = particleVelocities[Target] - particleVelocities[i]
          Force = Force * viscosityCoefficient * particleMass_2
          Force = Force / SpikySecondDerivative(Distance) / particleDensities[Target]
          particleExForces[i] += Force
    particleSumForces[i] = particleExForces[i]

@ti.kernel
def computeCollisions(Pred):
  for i in range(particleCnt):
    particleLocations[i,Pred][0] = min(max(particleLocations[i,Pred][0], boxX[0]), boxX[1])
    particleLocations[i,Pred][1] = min(max(particleLocations[i,Pred][1], boxY[0]), boxY[1])
    particleLocations[i,Pred][2] = min(max(particleLocations[i,Pred][2], boxZ[0]), boxZ[1])

@ti.kernel
def clearPressureAndForce():
  for i in range(particleCnt):
    particlePressures[i]  = 0
    particlePressureFs[i] = (0, 0, 0)    

@ti.kernel
def computePressure():
  for i in range(particleCnt):
    Error = max(particlePredictD[i] - Ro, 0)
    particlePressures[i] += max(Error * computeDelta(timeStep), 0)

@ti.kernel
def computePressureForces():
  for i in range(particleCnt):
    thisIndex = gridIndex(particleLocations[i])
    for k in range(27):
      newIndex = nextGrid(thisIndex, k)
      Idx = gridIndexInOne(newIndex)
      if not Idx == -1:
        for j in range(ti.static(maxParticlesPerGrid)):
          if j >= gridParticlesCnt[Idx]:
            break
          Target = gridParticles[Idx,j]
          Dir = particleLocations[Target] - particleLocations[i]
          A = particlePressures[i] / (particleDensities[i] * particleDensities[i])
          B = particlePressures[Target] / (particleDensities[Target] * particleDensities[Target])
          particlePressureFs[i] -= particleMass_2 * (A + B) * SpikyGradient(Dir)
    particleSumForces[i] = particleExForces[i] + particlePressureFs[i]

@ti.kernel
def Forward(Pred):
  for i in range(particleCnt):
    if Pred:
      predV = particleVelocities[i] + timeStep * particleSumForces[i] / particleMass
      particleLocations[i,1] = particleLocations[i,0] + predV * timeStep
    else:
      particleVelocities[i] += timeStep * particleSumForces[i] / particleMass
      particleLocations[i,0] += particleVelocities[i] * timeStep

@ti.kernel
def computeGrid(Pred):
  for i in range(gridCnt):
    gridParticles[i,0] = 0
  for i in range(particleCnt):
    Idx = gridIndexInOne(gridIndex(particleLocations[i,Pred]))
    setPlace = ti.atomic_add(gridParticles[Idx,0], 1)
    gridParticles[Idx,setPlace] = i

def Init():
  InitTaichi()

def Step():
  computeGrid(0)
  computeDensities(0)
  computeGravityForces()

  clearPressureAndForce()
  for i in range(maxPC):
    Forward(1)
    computeCollisions(1)
    computeGrid(1)
    computeDensities(1)
    if maxRo[None] < allowRoError + Ro:
      break
    computePressure()
    computePressureForces()

  Forward(0)
  computeCollisions(0)

def exportMesh(i: int):
  npL = particleLocations.to_numpy()

  mesh_writer = ti.PLYWriter(num_vertices=particleCnt, face_type="quad")
  mesh_writer.add_vertex_pos(npL[:, 0], npL[:, 1], npL[:, 2])
  mesh_writer.export_frame(i, 'S.ply')

def main():
  Init()
  frameT,frameIndex = 0,0
  try:
    while True:
      Step()
      frameT += timeStep
      if frameT > 1 / FPS:
        frameT -= 1 / FPS
        frameIndex += 1
        print('Frame', frameIndex)
        if Output:
          exportMesh(frameIndex)
  except Exception as error:
    print(error)

if __name__=='__main__':
  main()