eps = 1e-6

world_up = (0.0, 1.0, 0.0)

# the function has been deprecated
def calc_particle_radius(particle_mass: float, density: float) -> float:
    particle_cnt = int(density // particle_mass)
    particle_per_axis = int(particle_cnt ** (1/3) + 0.5)
    return 0.5 / particle_per_axis

def calc_particle_mass(particle_radius: float, density: float) -> float:
    particle_per_axis = 1 / particle_radius
    particle_cnt = particle_per_axis ** 3
    return density / particle_cnt