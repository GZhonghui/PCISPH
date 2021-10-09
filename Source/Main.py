import taichi as ti

ti.init(arch=ti.cuda)

tiPi = ti.acos(-1.0)

EPS = 1e-10

Output = True

timeStep = 0.00138889

minPC = 16
allowRoError = 0.1

FPS = 30

boxX = (-20.0, 20.0)
boxY = (-20.0, 20.0)
boxZ = ( 0.0,  15.0)

particleX = 60
particleY = 60
particleZ = 40

particleCnt = particleX * particleY * particleZ

particleMass = 0.00403914
particleMass_2 = particleMass * particleMass

viscosityCoefficient = 0

kernelR = 0.036
kernelR_2 = kernelR * kernelR
kernelR_3 = kernelR_2 * kernelR
kernelR_4 = kernelR_2 * kernelR_2
kernelR_5 = kernelR_2 * kernelR_3

Spacing = 0.02

searchR = Spacing * 1.2

gridX = (boxX[1] - boxX[0]) // (searchR * 2) + 1
gridY = (boxY[1] - boxY[0]) // (searchR * 2) + 1
gridZ = (boxZ[1] - boxZ[0]) // (searchR * 2) + 1

gridCnt = int(gridX * gridY * gridZ)

maxParticlesPerGrid = 128

Ro    = ti.field(dtype=ti.f32, shape=())
maxRo = ti.field(dtype=ti.f32, shape=())

Delta = ti.field(dtype=ti.f32, shape=())

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
  if Distance > EPS:
    Res = -SpikyFirstDerivative(Distance) * p / Distance
  return Res

@ti.kernel
def InitTaichi():
  for i in range(particleCnt):
    xIndex = (i %  particleX)
    yIndex = (i // particleX) % particleY
    zIndex = (i // (particleX * particleY))
    particleLocations[i,0][0] = -10 + Spacing * xIndex
    particleLocations[i,0][1] = -10 + Spacing * yIndex
    particleLocations[i,0][2] =  1  + Spacing * zIndex
    particleVelocities[i] = (0, 0, 0)

@ti.kernel
def computeDensities(Pred: ti.i32):
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
    maxRo[None] = max(maxRo[None], particleDensities[i,Pred])

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
          Distance = (particleLocations[i,0] - particleLocations[Target,0]).norm()
          Force = particleVelocities[Target] - particleVelocities[i]
          Force = Force * viscosityCoefficient * particleMass_2
          Force = Force / SpikySecondDerivative(Distance) / particleDensities[Target,0]
          particleExForces[i] += Force
    particleSumForces[i] = particleExForces[i]

@ti.kernel
def computeCollisions(Pred: ti.i32):
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
    Error = max(particleDensities[i,1] - Ro[None], 0)
    particlePressures[i] += max(Error * Delta[None], 0)

@ti.kernel
def computePressureForces():
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
          if particlePressures[i] > EPS or particlePressures[Target] > EPS:
            Dir = particleLocations[Target,0] - particleLocations[i,0]
            A = particlePressures[i] / (particleDensities[i,0] * particleDensities[i,0])
            B = particlePressures[Target] / (particleDensities[Target,0] * particleDensities[Target,0])
            particlePressureFs[i] -= particleMass_2 * (A + B) * SpikyGradient(Dir)
    particleSumForces[i] = particleExForces[i] + particlePressureFs[i]

@ti.kernel
def Forward(Pred: ti.i32):
  for i in range(particleCnt):
    if Pred:
      predV = particleVelocities[i] + timeStep * particleSumForces[i] / particleMass
      particleLocations[i,1] = particleLocations[i,0] + predV * timeStep
    else:
      particleVelocities[i] += timeStep * particleSumForces[i] / particleMass
      particleLocations[i,0] += particleVelocities[i] * timeStep

@ti.kernel
def computeGrid(Pred: ti.i32):
  for i in range(gridCnt):
    gridParticles[i,0] = 0
  for i in range(particleCnt):
    Idx = gridIndexInOne(gridIndex(particleLocations[i,Pred]))
    setPlace = ti.atomic_add(gridParticles[Idx,0], 1)
    gridParticles[Idx,setPlace+1] = i

@ti.kernel
def computeDelta():
  Beta = 2 * Square(particleMass * timeStep / Ro[None])
  xIdx,yIdx,zIdx = particleX // 2,particleY // 2,particleZ // 2
  partIdx = zIdx * particleX * particleY + yIdx * particleX + xIdx
  print('Detla Particle =',particleLocations[partIdx,0])
  thisIndex = gridIndex(particleLocations[partIdx,0])
  DemonA,DemonB = ti.Vector([0.0,0.0,0.0],dt=ti.f32),0.0
  for k in range(27):
    newIndex = nextGrid(thisIndex, k)
    Idx = gridIndexInOne(newIndex)
    if not Idx == -1:
      for j in range(1,maxParticlesPerGrid):
        if j > gridParticles[Idx,0]:
          break
        Target = gridParticles[Idx,j]
        gradWij = SpikyGradient(particleLocations[Target,0] - particleLocations[partIdx,0])
        DemonA += gradWij
        DemonB += gradWij.dot(gradWij)
  Demon = -DemonA.dot(DemonA) - DemonB
  if abs(Demon) > 0:
    Delta[None] = -1 / (Beta * Demon)
  else:
    Delta[None] = 0
  print('Delta =',Delta[None])

def Init():
  print('ParticleCnt = ',particleCnt)
  InitTaichi()
  computeGrid(0)
  computeDensities(0)
  Ro[None] = maxRo[None]
  print('Ro = ',Ro[None])
  computeDelta()

def Step():
  computeGrid(0)
  computeDensities(0)
  computeGravityForces()
  clearPressureAndForce()
  nowPC = 0
  while maxRo[None] > allowRoError + Ro[None] or nowPC < minPC:
    Forward(1)
    computeCollisions(1)
    computeGrid(1)
    computeDensities(1)
    print('PC (%03d): Max Densities ='%(nowPC),maxRo[None])
    computeGrid(0)
    computePressure()
    computePressureForces()
    nowPC += 1

  Forward(0)
  computeCollisions(0)

def exportMesh(i: int):
  npL = particleLocations.to_numpy()

  mesh_writer = ti.PLYWriter(num_vertices=particleCnt, face_type="quad")
  mesh_writer.add_vertex_pos(npL[:,0,0], npL[:,0,1], npL[:,0,2])
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

def test():
  Init()

if __name__=='__main__':
  test()