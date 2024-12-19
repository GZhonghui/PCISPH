import taichi as ti
import math
from Fluid._basic import *

@ti.data_oriented
class NeighborhoodSearcher:
    def __init__(self, parent):
        self.parent = parent
        self.domain_start = ti.math.vec3(0)
        self.domain_end = ti.math.vec3(1)
        self.grid_width = 0.1
        self.grid_cnt_per_axis = ti.math.ivec3(1)
        self.grid_cnt_sum = 1
        self.particles_cnt_in_every_grid = None

    def __del__(self):
        ...

    def init_grids(
            self, domain_start: list, domain_end: list,
            grid_width: float
        ):
        self.domain_start = ti.math.vec3(domain_start)
        self.domain_end = ti.math.vec3(domain_end)
        self.grid_width = grid_width
        self.grid_cnt_per_axis = ti.math.ivec3([
            math.ceil(domain_end[i] - domain_start[i] / grid_width)
            for i in range(3)
        ])
        for i in range(3):
            self.grid_cnt_sum *= self.grid_cnt_per_axis[i]
        log(f"space splited, grid count is {self.grid_cnt_sum}")

        self.particles_cnt_in_every_grid = ti.field(ti.int32)
        ti.root.dense(ti.i, self.grid_cnt_sum).place(
            self.particles_cnt_in_every_grid
        )

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_cnt_sum)

    @ti.func
    def location_to_grid_3d(
        self, location: ti.math.vec3 # type: ignore
    ) -> ti.math.ivec3: # type: ignore
        location = ti.math.clamp(location, self.domain_start, self.domain_end)
        return ti.cast((location - self.domain_start) / self.grid_width, ti.int32)

    @ti.func
    def grid_3d_to_grid_1d(self, grid_3d: ti.math.ivec3) -> int: # type: ignore
        return (
            grid_3d.x * self.grid_cnt_per_axis.y * self.grid_cnt_per_axis.z +
            grid_3d.y * self.grid_cnt_per_axis.z +
            grid_3d.z
        )

    @ti.kernel
    def clear_particles_cnt_in_every_grid(self):
        for i in range(self.grid_cnt_sum):
            self.particles_cnt_in_every_grid[i] = 0

    @ti.kernel
    def count_particles_cnt_in_every_grid(self, particles_cnt: int):
        for i in range(particles_cnt):
            grid_id = self.grid_3d_to_grid_1d(
                self.location_to_grid_3d(self.parent.particles[i].location)
            )
            ti.atomic_add(self.particles_cnt_in_every_grid[grid_id], 1)

    @ti.kernel
    def resort_particles(self, particles_cnt: int):
        for i in range(particles_cnt):
            grid_id = self.grid_3d_to_grid_1d(
                self.location_to_grid_3d(self.parent.particles[i].location)
            )
            id = ti.atomic_sub(self.particles_cnt_in_every_grid[grid_id], 1) - 1
            self.parent.particles[i] = id
            self.parent.id_to_index[id] = i

    def rebuild_search_index(self):
        self.clear_particles_cnt_in_every_grid()
        self.count_particles_cnt_in_every_grid(self.parent.particles_cnt)
        # self.prefix_sum_executor.run(self.particles_cnt_in_every_grid)
        # self.resort_particles(self.parent.particles_cnt)

    def for_all_neighborhood(self):
        ...