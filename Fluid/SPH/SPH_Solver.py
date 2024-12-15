import os
import taichi as ti
from Fluid._basic import *
from Fluid._importer._sph import *

class SPH_Solver:
    def __init__(self):
        self.cmd_args = dict()
        self.scene_cfg = dict()
        self.scene_data = dict()

    def __del__(self):
        ...

    def save_frame(self, idx: int) -> None:
        file_name = f"res_{idx:04}.txt"
        output_dir = os.path.abspath(self.cmd_args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        full_path = os.path.join(output_dir, file_name)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write("NULL")

    # Y axis is up
    def build_scene(self) -> bool:
        self.scene_cfg = load_scene(self.cmd_args.scene)

        if self.scene_cfg is None:
            log("build scene failed, because the scene file is missing")
            return False
        
        # read config complated
        # build data
        
        log("build scene complated")
        return True

    # simulation loop
    def run(self) -> None:
        log("running sph solver...")

        scene_parameters = self.scene_cfg["parameters"]
        total_steps = int(self.cmd_args.length // scene_parameters["time_step"])
        toal_frames = int(self.cmd_args.length * scene_parameters["frame_rate"])
        steps_per_frame = int(1 // (scene_parameters["frame_rate"] * scene_parameters["time_step"]))
        log(f"simulation length is {self.cmd_args.length} seconds")
        log(f"total simulation steps is {total_steps}")
        log(f"total output frames is {toal_frames}")
        log(f"output one frame after every {steps_per_frame} steps")

        frame_idx = 0
        for step_idx in range(total_steps):
            if step_idx % steps_per_frame == 0:
                self.save_frame(frame_idx)
                frame_idx += 1
            # simulation loop