import taichi as ti
import copy, math

eps = 1e-6 # do not use if you are not sure

speed_of_sound = 1433.0 # under water

world_up = (0.0, 1.0, 0.0)

eos_exponent = 7 # http://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf

# the function has been deprecated
def calc_particle_radius(particle_mass: float, density: float) -> float:
    particle_cnt = int(density // particle_mass)
    particle_per_axis = int(particle_cnt ** (1/3) + 0.5)
    return 0.5 / particle_per_axis

def calc_particle_mass(particle_radius: float, density: float) -> float:
    particle_per_axis = 1 / particle_radius
    particle_cnt = particle_per_axis ** 3
    return density / particle_cnt

# the function has been deprecated
def in_range(point: list, start: list, end: list) -> list:
    dim = len(point)
    in_point = [0] * dim
    is_in = True
    for i in range(dim):
        if point[i] < start[i] - eps or point[i] > end[i] + eps:
            is_in = False
        in_point[i] = math.min(math.max(point[i], start[i]), end[i])
    return [is_in, in_point]

kernel_func_h = 0.1
kernel_func_h2 = kernel_func_h ** 2
kernel_func_h3 = kernel_func_h ** 3
kernel_func_h4 = kernel_func_h ** 4
kernel_func_h5 = kernel_func_h ** 5

pi_h3_15 = 15 / (math.pi * kernel_func_h3)
pi_h4_45 = 45 / (math.pi * kernel_func_h4)
pi_h5_90 = 90 / (math.pi * kernel_func_h5)

def set_kernel_func_h(h: float):
    global kernel_func_h
    global kernel_func_h2, kernel_func_h3
    global kernel_func_h4, kernel_func_h5
    global pi_h3_15, pi_h4_45, pi_h5_90
    kernel_func_h = h
    kernel_func_h2 = h ** 2
    kernel_func_h3 = h ** 3
    kernel_func_h4 = h ** 4
    kernel_func_h5 = h ** 5
    pi_h3_15 = 15 / (math.pi * kernel_func_h3)
    pi_h4_45 = 45 / (math.pi * kernel_func_h4)
    pi_h5_90 = 90 / (math.pi * kernel_func_h5)

@ti.func
def spiky(r: float) -> float:
    result = 0.0
    if r < kernel_func_h:
        result = pi_h3_15 * ((1 - r / kernel_func_h) ** 3)
    return result

@ti.func
def spiky_first_derivative(r: float) -> float:
    result = 0.0
    if r < kernel_func_h:
        result = -pi_h4_45 * ((1 - r / kernel_func_h) ** 2)
    return result

@ti.func
def spiky_gradient(offset: ti.math.vec3) -> ti.math.vec3: # type: ignore
    result = 0.0
    distance = ti.math.length(offset)
    if distance > 0:
        result = -spiky_first_derivative(distance) * offset.normalized()
    return result

@ti.func
def spiky_second_derivative(r: float) -> float:
    result = 0.0
    if r < kernel_func_h:
        result = pi_h5_90 * (1 - r / kernel_func_h)
    return result

kernel_func = spiky
kernel_func_first_derivative = spiky_first_derivative
kernel_func_gradient = spiky_gradient
kernel_func_second_derivative = spiky_second_derivative