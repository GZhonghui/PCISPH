import os, json, math
import taichi as ti
from tqdm import tqdm
from Fluid._basic import *
from Fluid._importer._sph import *
from Fluid.SPH.NeighborhoodSearcher import NeighborhoodSearcher
from Fluid.SPH.ParticleSystem import ParticleSystem

class SPH_Solver:
    def __init__(self):
        self.cmd_args = None
        self.scene_cfg = None
        self.particle_system = ParticleSystem()
        self.neighborhood_searcher = NeighborhoodSearcher()

    def __del__(self):
        ...

    def save_frame(self, idx: int) -> None:
        file_name = f"res_{idx:04}.json"
        output_dir = os.path.abspath(self.cmd_args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        particles = list()
        self.particle_system.export_particles_location(particles)

        full_path = os.path.join(output_dir, file_name)
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump({"particles": particles}, file, ensure_ascii=False, indent=4)

    @log_time
    def build_scene(self) -> bool:
        self.scene_cfg = load_scene(self.cmd_args.scene)

        if self.scene_cfg is None:
            log("build scene failed, because the scene file is missing")
            return False
        
        # read config complated
        # build data
        parameters = self.scene_cfg["parameters"]
        particle_mass, density = parameters["particle_mass"], parameters["density"]
        particle_radius = calc_radius(particle_mass, density)
        log(f"particle calc complated, particle radius is {particle_radius}")

        particles = list()
        fluid_blocks = self.scene_cfg["fluid_blocks"]
        for fluid_block in fluid_blocks:
            domain_start = fluid_block["domain_start"]
            domain_end = fluid_block["domain_end"]
            x_pos = domain_start[0] + particle_radius
            while x_pos < domain_end[0] + eps:
                y_pos = domain_start[1] + particle_radius
                while y_pos < domain_end[1] + eps:
                    z_pos = domain_start[2] + particle_radius
                    while z_pos < domain_end[2] + eps:
                        particles.append([x_pos, y_pos, z_pos])
                        z_pos += particle_radius * 2
                    y_pos += particle_radius * 2
                x_pos += particle_radius * 2
        particles_cnt = len(particles)

        self.particle_system.malloc_memory(particles_cnt)
        for idx in range(particles_cnt):
            self.particle_system.set_particle_location(idx, particles[idx])

        log(f"build scene complated, particle count is {particles_cnt:,}")
        return True

    # simulation loop
    @log_time
    def run(self) -> None:
        log("running sph solver...")

        scene_parameters = self.scene_cfg["parameters"]
        total_steps = int(self.cmd_args.length // scene_parameters["time_step"])
        steps_per_frame = math.ceil(1 / (scene_parameters["frame_rate"] * scene_parameters["time_step"]))
        toal_frames = math.ceil(total_steps / steps_per_frame)
        
        log(f"simulation length is {self.cmd_args.length} seconds")
        log(f"total simulation steps is {total_steps}")
        log(f"total output frames is about {toal_frames}")
        log(f"output one frame after every {steps_per_frame} steps")

        frame_idx = 0
        enter_bar()
        for step_idx in tqdm(range(total_steps)):
            if step_idx % steps_per_frame == 0:
                self.save_frame(frame_idx)
                frame_idx += 1
            # simulation loop
        exit_bar()

        log("sph solver run complated")