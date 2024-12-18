import taichi as ti

# @ti.dataclass
Particle = ti.types.struct(
    id = ti.int32,
    location = ti.math.vec3,
    density = ti.math.vec3,
    forces = ti.math.vec3,
)