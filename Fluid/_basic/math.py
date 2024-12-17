eps = 1e-6

world_up = (0.0, 1.0, 0.0)

def calc_radius(particle_mass: float, density: float) -> float:
    particle_cnt = int(density // particle_mass)
    particle_per_axis = int(particle_cnt ** (1/3) + 0.5)
    return 0.5 / particle_per_axis