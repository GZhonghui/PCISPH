import taichi as ti

# @ti.dataclass
Particle = ti.types.struct(
    id = ti.int32,
    grid_id = ti.int32,
    location = ti.math.vec3,
    density = ti.f32,
    pressure = ti.f32,
    forces = ti.math.vec3,
    pressure_forces = ti.math.vec3,
    velocity = ti.math.vec3
)