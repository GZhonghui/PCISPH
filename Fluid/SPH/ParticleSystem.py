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
        self.domain_start = ti.math.vec3(domain_start)
        self.domain_end = ti.math.vec3(domain_end)

    def init_parameters(
        self,
        particle_radius: float,
        particle_mass: float,
        density: float,
        gravitation: list,
        viscosity_coefficient: float,
        time_step: float
    ):
        self.particle_radius = particle_radius
        self.particle_mass = particle_mass
        self.density = density
        self.gravitation = ti.math.vec3(gravitation)
        self.viscosity_coefficient = viscosity_coefficient
        self.time_step = time_step

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

    # TODO: check
    @ti.func
    def add_viscosity_force(self, self_index: int, other_index: int):
        distance = ti.math.distance(
            self.particles[self_index].location,
            self.particles[other_index].location
        )
        self.particles[self_index].forces += (
            (self.particles[other_index].location - self.particles[self_index].location)
            * self.viscosity_coefficient * self.particle_mass * self.particle_mass
            * kernel_func_second_derivative(distance)
            / self.particles[other_index].density
        )

    # TODO: check
    @ti.kernel
    def accumulate_viscosity_force(self):
        for i in range(self.particles_cnt):
            self.neighborhood_searcher.for_all_neighborhoods(i, self.add_viscosity_force)

    # TODO: check
    @ti.func
    def compute_pressure_from_eos(self, density: float, eos_scale: float):
        pressure = eos_scale * (
            ti.math.pow(density / self.density, eos_exponent) - 1.0
        )
        
        # TODO: is this right?
        if pressure < 0:
            pressure = 0
        
        return pressure

    # TODO: check
    @ti.kernel
    def compute_pressure(self):
        eos_scale = self.density * speed_of_sound * speed_of_sound / eos_exponent
        for i in range(self.particles_cnt):
            self.particles[i].pressure = self.compute_pressure_from_eos(
                self.particles[i].density,
                eos_scale
            )

    # TODO: check
    @ti.func
    def add_pressure_force(self, self_index: int, other_index: int):
        # distance = ti.math.distance(
        #     self.particles[self_index].location,
        #     self.particles[other_index].location
        # )
        if self_index != other_index:
            # dir = (
            #     self.particles[other_index].location - self.particles[self_index].location
            # ) / distance
            part_self = self.particles[self_index].pressure / (
                self.particles[self_index].density * self.particles[self_index].density
            )
            part_other = self.particles[other_index].pressure / (
                self.particles[other_index].density * self.particles[other_index].density
            )
            self.particles[self_index].pressure_forces -= (
                self.particle_mass * self.particle_mass
                * (part_self + part_other)
                / kernel_func_gradient(self.particles[other_index].location - self.particles[self_index].location)
            )

    # TODO: check
    @ti.kernel
    def accumulate_pressure_force(self):
        for i in range(self.particles_cnt):
            self.particles[i].pressure_forces = ti.math.vec3(0)
            self.neighborhood_searcher.for_all_neighborhoods(i, self.add_pressure_force)
            self.particles[i].forces += self.particles[i].pressure_forces

    @ti.kernel
    def time_integration(self):
        for i in range(self.particles_cnt):
            self.particles[i].velocity += (
                self.particles[i].forces * self.time_step / self.particle_mass
            )
            self.particles[i].location += self.particles[i].velocity * self.time_step

    # TODO: is this right?
    @ti.kernel
    def resolve_collision(self):
        for i in range(self.particles_cnt):
            location = self.particles[i].location
            if (
                location.x < self.domain_start.x
                or location.x > self.domain_end.x
                or location.y < self.domain_start.y
                or location.y > self.domain_end.y
                or location.z < self.domain_start.z
                or location.z > self.domain_end.z
            ):
                self.particles[i].velocity *= -0.5
                self.particles[i].location = ti.math.clamp(
                    location, self.domain_start, self.domain_end
                )