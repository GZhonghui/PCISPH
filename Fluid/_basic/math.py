import taichi as ti
import copy, math

eps = 1e-6 # do not use if you are not sure

# 水中的声速
speed_of_sound = 1433.0 # under water

# 世界坐标系的上方向
world_up = (0.0, 1.0, 0.0)

# 状态方程的指数
eos_exponent = 7 # http://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf

# the function has been deprecated
def calc_particle_radius(particle_mass: float, density: float) -> float:
    particle_cnt = int(density // particle_mass)
    particle_per_axis = int(particle_cnt ** (1/3) + 0.5)
    return 0.5 / particle_per_axis

# 计算粒子质量
def calc_particle_mass(particle_radius: float, density: float) -> float:
    particle_per_axis = 0.5 / particle_radius # 0.5 / 半径 = 单个轴上的粒子数量
    particle_cnt = particle_per_axis ** 3 # 单个轴上的粒子数量 ^ 3 = 总粒子数量
    return density / particle_cnt # 密度（单位体积内的粒子质量） / 单位体积内的总粒子数量 = 单个粒子的质量

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

def get_kernel_func_h():
    return kernel_func_h

kernel_func_h2 = 0
kernel_func_h3 = 0
kernel_func_h4 = 0
kernel_func_h5 = 0

pi_h3_15 = 0
pi_h4_45 = 0
pi_h5_90 = 0

pi_h3_64_315 = 0
pi_h5_32_945 = 0

def set_kernel_func_h(h: float):
    global kernel_func_h
    global kernel_func_h2
    global kernel_func_h3
    global kernel_func_h4
    global kernel_func_h5
    global pi_h3_15
    global pi_h4_45
    global pi_h5_90
    global pi_h3_64_315
    global pi_h5_32_945

    kernel_func_h = h
    kernel_func_h2 = h ** 2
    kernel_func_h3 = h ** 3
    kernel_func_h4 = h ** 4
    kernel_func_h5 = h ** 5
    pi_h3_15 = 15 / (math.pi * kernel_func_h3)
    pi_h4_45 = 45 / (math.pi * kernel_func_h4)
    pi_h5_90 = 90 / (math.pi * kernel_func_h5)

    pi_h3_64_315 = 315 / (64 * math.pi * kernel_func_h3)
    pi_h5_32_945 = 945 / (32 * math.pi * kernel_func_h5)

set_kernel_func_h(kernel_func_h)

@ti.func
def poly6(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        x = 1 - r * r / kernel_func_h2
        result = pi_h3_64_315 * x * x * x
    return result

@ti.func
def poly6_first_derivative(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        x = 1 - r * r / kernel_func_h2
        result = -pi_h5_32_945 * r * x * x
    return result

@ti.func
def poly6_gradient(offset_point_to_center: ti.math.vec3) -> ti.math.vec3: # type: ignore
    result = ti.math.vec3(0.0)
    distance = ti.math.length(offset_point_to_center)
    if 0.0 < distance and distance < kernel_func_h:
        result = poly6_first_derivative(distance) * offset_point_to_center.normalized()
    return result

@ti.func
def poly6_second_derivative(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        x = r * r / kernel_func_h2
        result = pi_h5_32_945 * (1 - x) * (5 * x - 1)
    return result

@ti.func
def spiky(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        result = pi_h3_15 * ((1 - r / kernel_func_h) ** 3)
    return result

@ti.func
def spiky_first_derivative(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        result = -pi_h4_45 * ((1 - r / kernel_func_h) ** 2)
    return result

# 注意参数是指向中心点的向量
@ti.func
def spiky_gradient(offset_point_to_center: ti.math.vec3) -> ti.math.vec3: # type: ignore
    result = ti.math.vec3(0.0)
    distance = ti.math.length(offset_point_to_center)
    if 0.0 < distance and distance < kernel_func_h:
        result = spiky_first_derivative(distance) * offset_point_to_center.normalized()
    return result

@ti.func
def spiky_second_derivative(r: float) -> float:
    result = 0.0
    if not r < 0.0 and r < kernel_func_h:
        result = pi_h5_90 * (1 - r / kernel_func_h)
    return result

kernel_func_a = spiky
kernel_func_a_first_derivative = spiky_first_derivative
kernel_func_a_gradient = spiky_gradient
kernel_func_a_second_derivative = spiky_second_derivative

kernel_func_b = poly6
kernel_func_b_first_derivative = poly6_first_derivative
kernel_func_b_gradient = poly6_gradient
kernel_func_b_second_derivative = poly6_second_derivative
