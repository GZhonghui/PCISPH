import taichi as ti

# @ti.dataclass
Particle = ti.types.struct(
    id = ti.uint32,
    location = ti.math.vec3
)