import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

tiPi = ti.acos(-1.0)

Output = True

timeStep = 0.001

maxPC = 10
allowError = 0.1

FPS = 30

boxX = (-40.0, 40.0)
boxY = (-40.0, 40.0)
boxZ = ( 0.0,  30.0)

particleCnt = 120 * 120 * 80

particleMass = 0.001
particleMass_2 = particleMass * particleMass

Ro = 1.4147106

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

Gravity = ti.Vector([0.0, 0.0, -9.8], dt=ti.f32)

particleLocations  = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particleDensities  = ti.field(dtype=ti.f32, shape=particleCnt)
particleExForces   = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particleVelocities = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particelPressure   = ti.field(dtype=ti.f32, shape=particleCnt)
particelPressureF  = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particelSumForce   = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particlePredictD   = ti.field(dtype=ti.f32, shape=particleCnt)
particlePredictL   = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)
particlePredictV   = ti.Vector.field(3, dtype=ti.f32, shape=particleCnt)

gridParticlesIdx = ti.Vector.field(3, dtype=ti.i32, shape=particleCnt)
gridParticlesCnt = ti.field(dtype=ti.i32, shape=gridCnt)
gridParticles    = ti.field(dtype=ti.i32, shape=(gridCnt, maxParticlesPerGrid))

maxRo = ti.field(dtype=ti.f32, shape=())

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

@ti.kernel
def InitTaichi():
    for i in particleLocations:
        xIndex = (i %  120)
        yIndex = (i // 120) % 120
        zIndex = (i // (120 * 120))
        particleLocations[i][0] = -10 + kernelR * 1.01 * xIndex
        particleLocations[i][1] = -10 + kernelR * 1.01 * yIndex
        particleLocations[i][2] =  15 + kernelR * 1.01 * zIndex

@ti.kernel
def computeDensities():
    for i in particleDensities:
        Sum = 0.0
        thisIndex = gridParticlesIdx[i]
        for k in range(27):
            newIndex = nextGrid(thisIndex, k)
            Idx = gridIndexInOne(newIndex)
            if not Idx == -1:
                for j in range(ti.static(maxParticlesPerGrid)):
                    if j >= gridParticlesCnt[Idx]:
                        break
                    Target = gridParticles[Idx,j]
                    D = (particleLocations[i] - particleLocations[Target]).norm()
                    Sum += Spiky(D)
        particleDensities[i] = particleMass * Sum

@ti.kernel
def computePredictD():
    for i in particlePredictD:
        Sum = 0.0
        thisIndex = gridParticlesIdx[i]
        for k in range(27):
            newIndex = nextGrid(thisIndex, k)
            Idx = gridIndexInOne(newIndex)
            if not Idx == -1:
                for j in range(ti.static(maxParticlesPerGrid)):
                    if j >= gridParticlesCnt[Idx]:
                        break
                    Target = gridParticles[Idx,j]
                    D = (particlePredictL[i] - particlePredictL[Target]).norm()
                    Sum += Spiky(D)
        particlePredictD[i] = particleMass * Sum
        maxRo[None] = max(maxRo[None], particlePredictD[i])

@ti.kernel
def clearExForces():
    for i in particleExForces:
        particleExForces[i] = (0, 0, 0)

@ti.kernel
def computeExternalForces():
    for i in particleExForces:
        particleExForces[i] += Gravity

@ti.kernel
def computeViscosityForces():
    for i in particleExForces:
        thisIndex = gridParticlesIdx[i]
        for k in range(27):
            newIndex = nextGrid(thisIndex, k)
            Idx = gridIndexInOne(newIndex)
            if not Idx == -1:
                for j in range(ti.static(maxParticlesPerGrid)):
                    if j >= gridParticlesCnt[Idx]:
                        break
                    Target = gridParticles[Idx,j]
                    Distance = (particleLocations[i] - particleLocations[Target]).norm()
                    Force = particleVelocities[Target] - particleVelocities[i]
                    Force = Force * viscosityCoefficient * particleMass_2
                    Force = Force / SpikySecondDerivative(Distance) / particleDensities[Target]

                    particleExForces[i] += Force

@ti.kernel
def computeCollisions():
    for i in particleLocations:
        particleLocations[i][0] = min(max(particleLocations[i][0], boxX[0]), boxX[1])
        particleLocations[i][1] = min(max(particleLocations[i][1], boxY[0]), boxY[1])
        particleLocations[i][2] = min(max(particleLocations[i][2], boxZ[0]), boxZ[1])

@ti.kernel
def computePredictCollisions():
    for i in particlePredictL:
        particlePredictL[i][0] = min(max(particlePredictL[i][0], boxX[0]), boxX[1])
        particlePredictL[i][1] = min(max(particlePredictL[i][1], boxY[0]), boxY[1])
        particlePredictL[i][2] = min(max(particlePredictL[i][2], boxZ[0]), boxZ[1])

@ti.func
def computeBeta(t):
    return 2 * Square(particleMass * t * Ro)

@ti.func
def computeDelta(t):
    return 1

@ti.kernel
def clearPressure():
    for i in particelPressure:
        particelPressure[i] = 0

@ti.kernel
def computePressure():
    for i in particelPressure:
        Error = max(particlePredictD[i] - Ro, 0)
        particelPressure[i] += max(Error * computeDelta(timeStep), 0)

@ti.kernel
def clearPressureF():
    for i in particelPressureF:
        particelPressureF[i] = (0, 0, 0)

@ti.kernel
def computePressureForces():
    for i in particelPressure:
        thisIndex = gridParticlesIdx[i]
        for k in range(27):
            newIndex = nextGrid(thisIndex, k)
            Idx = gridIndexInOne(newIndex)
            if not Idx == -1:
                for j in range(ti.static(maxParticlesPerGrid)):
                    if j >= gridParticlesCnt[Idx]:
                        break
                    Target = gridParticles[Idx,j]
                    Dir = particleLocations[Target] - particleLocations[i]
                    A = particelPressure[i] / (particleDensities[i] * particleDensities[i])
                    B = particelPressure[Target] / (particleDensities[Target] * particleDensities[Target])
                    particelPressureF[i] -= particleMass_2 * (A + B) * SpikyGradient(Dir)
        particelSumForce[i] = particleExForces[i] + particelPressureF[i]

@ti.kernel
def computePredictVL():
    for i in particlePredictV:
        particlePredictV[i] = particleVelocities[i] + timeStep * particelSumForce[i] / particleMass
    for i in particlePredictL:
        particlePredictL[i] = particleLocations[i] + particlePredictV[i] * timeStep

@ti.kernel
def clearVelocities():
    for i in particleVelocities:
        particleVelocities[i] = (0, 0, 0)

@ti.kernel
def computeVelocities():
    for i in particleVelocities:
        particleVelocities[i] += timeStep * particelSumForce[i] / particleMass

@ti.kernel
def computeLocation():
    for i in particleLocations:
        particleLocations[i] += particleVelocities[i] * timeStep

@ti.kernel
def clearGrid():
    for i in gridParticlesCnt:
        gridParticlesCnt[i] = 0

@ti.kernel
def computeGrid():
    for i in particleLocations:
        gridParticlesIdx[i] = gridIndex(particleLocations[i])
        Idx = gridIndexInOne(gridParticlesIdx[i])

        setPlace = ti.atomic_add(gridParticlesCnt[Idx], 1)
        gridParticles[Idx, setPlace] = i

@ti.kernel
def computeGridPredict():
    for i in particlePredictL:
        gridParticlesIdx[i] = gridIndex(particlePredictL[i])
        Idx = gridIndexInOne(gridParticlesIdx[i])

        setPlace = ti.atomic_add(gridParticlesCnt[Idx], 1)
        gridParticles[Idx, setPlace] = i

def Init():
    InitTaichi()

    clearVelocities()

def Step():
    clearGrid()
    computeGrid()

    computeDensities()

    clearExForces()
    computeExternalForces()
    computeViscosityForces()

    clearPressure()
    clearPressureF()

    for i in range(maxPC):
        computePressureForces()

        computePredictVL()
        computePredictCollisions()

        clearGrid()
        computeGridPredict()

        computePredictD()

        if maxRo[None] < allowError + Ro:
            break
        
        computePressure()

    computeVelocities()
    computeLocation()
    computeCollisions()

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