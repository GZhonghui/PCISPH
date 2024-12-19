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
    
    @ti.func
    def grid_1d_to_grid_3d(self, grid_1d: int) -> tuple:
        x = int(grid_1d / (self.grid_cnt_per_axis.y * self.grid_cnt_per_axis.z))
        grid_1d -= x
        y = int(grid_1d / self.grid_cnt_per_axis.z)
        grid_1d -= y
        z = grid_1d
        return x,y,z

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
            self.parent.particles[i].id = id
            self.parent.particles[i].grid_id = grid_id
            self.parent.id_to_index[id] = i

    # rebuild every steps
    def rebuild_search_index(self):
        self.clear_particles_cnt_in_every_grid()
        self.count_particles_cnt_in_every_grid(self.parent.particles_cnt)
        # cant run on arm64
        # self.prefix_sum_executor.run(self.particles_cnt_in_every_grid)
        # self.resort_particles(self.parent.particles_cnt)

    # call_func(self_index, other_index)
    @ti.func
    def for_all_neighborhoods(self, index: int, call_func: ti.template()): # type: ignore
        grid_id = self.parent.particles[index].grid_id
        x,y,z = self.grid_1d_to_grid_3d(grid_id)
        # neighbors = [
        #     [x+i, y+j, z+k] 
        #     for i in range(-1,2)
        #     for j in range(-1,2)
        #     for k in range(-1,2)
        # ]
        # for neighbor_idx in range(27):
        #     x,y,z = neighbors[neighbor_idx]
        #     if (
        #         0 <= x < self.grid_cnt_per_axis.x and
        #         0 <= y < self.grid_cnt_per_axis.y and
        #         0 <= z < self.grid_cnt_per_axis.z
        #     ):
        #         neighbor_id = self.grid_3d_to_grid_1d(ti.math.ivec3(x,y,z))
        #         l, r  = 0, self.particles_cnt_in_every_grid[neighbor_id]
        #         if neighbor_id > 0:
        #             l = self.particles_cnt_in_every_grid[neighbor_id - 1]
        #         for i in range(l,r):
        #             call_func(index, self.parent.id_to_index[i])

        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            neighbor = ti.math.ivec3(offset) + ti.math.ivec3(x,y,z)
            x,y,z = neighbor.x, neighbor.y, neighbor.z
            if (
                0 <= x < self.grid_cnt_per_axis.x and
                0 <= y < self.grid_cnt_per_axis.y and
                0 <= z < self.grid_cnt_per_axis.z
            ):
                neighbor_id = self.grid_3d_to_grid_1d(ti.math.ivec3(x,y,z))
                l, r  = 0, self.particles_cnt_in_every_grid[neighbor_id]
                if neighbor_id > 0:
                    l = self.particles_cnt_in_every_grid[neighbor_id - 1]
                for i in range(l,r):
                    call_func(index, self.parent.id_to_index[i])