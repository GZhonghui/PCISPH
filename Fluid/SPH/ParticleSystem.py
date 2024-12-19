import taichi as ti

from Fluid._basic import *
from Fluid.SPH.Particle import Particle
from Fluid.SPH.NeighborhoodSearcher import NeighborhoodSearcher

@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        self.particles_cnt = 0
        self.particles = None
        self.id_to_index = None # TODO
        self.neighborhood_searcher = NeighborhoodSearcher(self)

    def __del__(self):
        ...

    @ti.kernel
    def init_memory(self):
        for idx in range(self.particles_cnt):
            self.particles[idx].id = ti.int32(idx)
            self.id_to_index[idx] = ti.int32(idx)

    def malloc_memory(self, particles_cnt: int):
        self.particles_cnt = particles_cnt
        # self.particles = Particle.field(shape=(particles_cnt,))
        self.particles = Particle.field()
        ti.root.dense(ti.i, particles_cnt).place(self.particles)
        self.id_to_index = ti.field(ti.int32)
        ti.root.dense(ti.i, particles_cnt).place(self.id_to_index)
        # self.particles_location_field = ti.Vector.field(
        #     3, dtype=ti.f32, shape = particles_cnt
        # )
        self.init_memory()

    # too slow...
    # the function has been deprecated
    def set_particle_location(self, id: int, location: list):
        if id < 0 or id >= self.particles_cnt:
            log("particle id out of range")
            return
        self.particles[id].location = ti.math.vec3(location)

    @ti.kernel
    def init_particles_location_of_one_fluid_block(
        self,
        start_x: float, start_y: float, start_z: float,
        cnt_x: int, cnt_y: int, cnt_z: int,
        inited_cnt: int,
        particle_radius: float
    ):
        for i,j,k in ti.ndrange(cnt_x, cnt_y, cnt_z):
            id = inited_cnt + i * cnt_y * cnt_z + j * cnt_z + k
            self.particles[id].location = ti.math.vec3(
                start_x + particle_radius * (i * 2 + 1),
                start_y + particle_radius * (j * 2 + 1),
                start_z + particle_radius * (k * 2 + 1)
            )

    def init_particles_location(
            self, fluid_blocks_expand: list, particle_radius: float
        ):
        inited_cnt = 0
        for fluid_block in fluid_blocks_expand:
            self.init_particles_location_of_one_fluid_block(
                *fluid_block["start"],
                *fluid_block["cnt"],
                inited_cnt,
                particle_radius
            )
            inited_cnt += fluid_block["sum"]

    def export_particles_location_to_list(self, location_list: list):
        location_list.clear()
        for idx in range(self.particles_cnt):
            location = self.particles[idx].location
            location_list.append([
                location.x, location.y, location.z
            ])
    
    # the function has been deprecated
    @ti.kernel
    def export_particles_location_to_field(self):
        for idx in range(self.particles_cnt):
            self.particles_location_field[idx] = self.particles[idx].location

    def init_domain(
        self,
        domain_start: list,
        domain_end: list,
        grid_width: float
    ):
        self.neighborhood_searcher.init_grids(
            domain_start,
            domain_end,
            grid_width
        )

    def init_parameters(
        self,
        particle_radius: float,
        particle_mass: float,
        density: float,
        gravitation: list
    ):
        self.particle_radius = particle_radius
        self.particle_mass = particle_mass
        self.density = density
        self.gravitation = ti.math.vec3(gravitation)

    def rebuild_search_index(self):
        self.neighborhood_searcher.rebuild_search_index()

    @ti.func
    def add_density(self, self_index: int, other_index: int):
        distance = ti.math.distance(
            self.particles[self_index].location,
            self.particles[other_index].location
        )
        self.particles[self_index].density += kernel_func(distance)

    @ti.kernel
    def compute_densities(self):
        for i in range(self.particles_cnt):
            self.particles[i].density = 0
            self.neighborhood_searcher.for_all_neighborhoods(i, self.add_density)
            self.particles[i].density *= self.particle_mass

    # TODO: wind forces
    @ti.kernel
    def accumulate_external_forces(self):
        for i in range(self.particles_cnt):
            self.particles[i].forces = self.particle_mass * self.gravitation

    @ti.kernel
    def accumulate_viscosity_force(self):
        ...

    @ti.kernel
    def compute_pressure(self):
        ...

    @ti.kernel
    def accumulate_pressure_force(self):
        ...

    @ti.kernel
    def time_integration(self):
        ...

    @ti.kernel
    def resolve_collision(self):
        ...