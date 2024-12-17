import taichi as ti

from Fluid._basic import *
from Fluid.SPH.Particle import Particle

@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        self.particles = None
        self.particles_cnt = 0

    def __del__(self):
        ...

    @ti.kernel
    def init_memory(self):
        for idx in range(self.particles_cnt):
            self.particles[idx].id = ti.uint32(idx)

    def malloc_memory(self, particles_cnt: int):
        self.particles_cnt = particles_cnt
        self.particles = Particle.field(shape=(particles_cnt,))
        self.init_memory()

    def set_particle_location(self, id: int, location: list):
        if id < 0 or id >= self.particles_cnt:
            log("particle id out of range")
            return
        self.particles[id].location = ti.math.vec3(location)

    def export_particles_location(self, location_list: list):
        location_list.clear()
        for idx in range(self.particles_cnt):
            location = self.particles[idx].location
            location_list.append([
                location.x, location.y, location.z
            ])