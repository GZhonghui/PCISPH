import os, json, math, copy
import imageio
import numpy as np
import taichi as ti
from tqdm import tqdm
from Fluid._basic import *
from Fluid.SPH.ParticleSystem import ParticleSystem

class SPH_Solver:
    def __init__(self):
        self.cmd_args = None
        self.scene_cfg = None
        self.particle_system = ParticleSystem()

    def __del__(self):
        ...

    # save particles location to disk
    def save_frame(self, idx: int) -> None:
        if not self.cmd_args.enable_output:
            return
        
        file_name = f"res_{idx:04}.json"
        
        particles = list()
        self.particle_system.export_particles_location_to_list(particles)

        full_path = os.path.join(self.output_dir, file_name)
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
        particle_radius = parameters["particle_radius"]
        density = parameters["density"]
        particle_mass = calc_particle_mass(particle_radius, density)
        log(f"particle calc complated, particle mass is {particle_mass}")

        # set kernel function h
        kernel_func_h = particle_radius * 4.0
        set_kernel_func_h(kernel_func_h)
        log(f"set kernel function h to {kernel_func_h}")

        gravitation = parameters["gravitation"]
        viscosity_coefficient = parameters["viscosity_coefficient"]
        time_step = parameters["time_step"]
        # init particle parameters
        self.particle_system.init_parameters(
            particle_radius,
            particle_mass,
            density,
            gravitation,
            viscosity_coefficient,
            time_step,
            kernel_func_h
        )

        # init grid
        domain_start = parameters["domain_start"]
        domain_end = parameters["domain_end"]
        self.particle_system.init_domain(
            domain_start, domain_end,
            grid_width=kernel_func_h
        )

        fluid_blocks = self.scene_cfg["fluid_blocks"]
        # particles = list()
        # for fluid_block in fluid_blocks:
        #     domain_start = fluid_block["domain_start"]
        #     domain_end = fluid_block["domain_end"]
        #     x_pos = domain_start[0] + particle_radius
        #     while x_pos < domain_end[0] + eps:
        #         y_pos = domain_start[1] + particle_radius
        #         while y_pos < domain_end[1] + eps:
        #             z_pos = domain_start[2] + particle_radius
        #             while z_pos < domain_end[2] + eps:
        #                 particles.append([x_pos, y_pos, z_pos])
        #                 z_pos += particle_radius * 2
        #             y_pos += particle_radius * 2
        #         x_pos += particle_radius * 2
        # particles_cnt = len(particles)

        fluid_blocks_expand = list()
        particles_cnt = 0
        for fluid_block in fluid_blocks:
            domain_start = fluid_block["domain_start"]
            domain_end = fluid_block["domain_end"]
            cnt_per_axis = [0] * 3
            cnt_in_this_block = 1
            for i in range(3):
                cnt_per_axis[i] = math.ceil(
                    (domain_end[i] - domain_start[i]) / (particle_radius * 2)
                )
                cnt_in_this_block *= cnt_per_axis[i]
            particles_cnt += cnt_in_this_block
            fluid_blocks_expand.append({
                "start": domain_start,
                "cnt": cnt_per_axis,
                "sum": cnt_in_this_block
            })

        log("start malloc data on computing device")
        self.particle_system.malloc_memory(particles_cnt)

        log("initing particles location...")
        self.particle_system.init_particles_location(
            fluid_blocks_expand, particle_radius
        )

        # for idx in range(particles_cnt):
        #     self.particle_system.set_particle_location(idx, particles[idx])

        log(f"build scene complated, particle count is {particles_cnt:,}")
        return True
    
    def step(self):
        self.particle_system.rebuild_search_index()
        self.particle_system.compute_densities()
        self.particle_system.accumulate_external_forces()
        # self.particle_system.accumulate_viscosity_force()
        self.particle_system.compute_pressure()
        self.particle_system.accumulate_pressure_force()
        self.particle_system.time_integration()
        self.particle_system.resolve_collision()

    # simulation loop
    @log_time
    def run(self) -> None:
        log("running sph solver...")

        scene_parameters = self.scene_cfg["parameters"]
        particle_radius = scene_parameters["particle_radius"]
        total_steps = int(self.cmd_args.length // scene_parameters["time_step"])
        steps_per_frame = math.ceil(
            1 / (scene_parameters["frame_rate"] * scene_parameters["time_step"])
        )
        toal_frames = math.ceil(total_steps / steps_per_frame)
        
        log(f"simulation length is {self.cmd_args.length} seconds")
        log(f"total simulation steps is {total_steps}")
        log(f"total output frames is about {toal_frames}")
        log(f"output one frame after every {steps_per_frame} steps")

        self.output_dir = os.path.abspath(self.cmd_args.output_path)
        os.makedirs(self.output_dir, exist_ok=True)

        enable_preview = self.cmd_args.enable_preview

        frame_rate = scene_parameters["frame_rate"]
        render_cfg = self.scene_cfg["render"]
        width, height = render_cfg["width"], render_cfg["height"]
        if enable_preview:
            self.preview_window = ti.ui.Window(
                name="Preview",
                res=(width,height),
                fps_limit=frame_rate,
                pos=(128,128),
                show_window=False
            )

            self.video_manager = ti.tools.VideoManager(
                self.output_dir,
                # video_filename="rendered_by_vulkan"
            )

            frame_images = []

            # render
            canvas = self.preview_window.get_canvas()
            scene = self.preview_window.get_scene()
            camera = ti.ui.Camera()
            camera.position(*render_cfg["camera_location"])
            camera.lookat(*render_cfg["camera_target"])
            camera.up(*world_up)
            camera.fov(render_cfg["camera_fov"])
            camera.projection_mode(ti.ui.ProjectionMode.Perspective)
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))

        log("running neighborhood search debug...")
        self.particle_system.rebuild_search_index()
        self.particle_system.run_neighborhood_search_debug()

        frame_idx = 0
        enter_bar()
        pbar = tqdm(total=total_steps, desc="simulation")
        for step_idx in range(total_steps):
            # one frame
            if step_idx % steps_per_frame == 0:
                # export particles location to disk
                self.save_frame(frame_idx)
                frame_idx += 1

                # update preview window
                if enable_preview and self.preview_window.running:
                    scene.particles(
                        self.particle_system.particles.location,
                        color = (0.68, 0.26, 0.19),
                        radius = particle_radius
                    )
                    canvas.scene(scene)
                    # self.preview_window.show()
                    image = self.preview_window.get_image_buffer_as_numpy()
                    frame_images.append(image)
                    # ti.tools.imwrite(
                    #     image,
                    #     os.path.join(self.output_dir, "test.png")
                    # )
                    # log(f"write image to {os.path.join(self.output_dir, 'test.png')}")
                    # self.video_manager.write_frame(image)
            # simulation loop body
            self.step()
            pbar.set_postfix_str(f"AD: {self.particle_system.compute_avg_density():.2f}")
            pbar.update(1)
        pbar.close()
        exit_bar()
        if enable_preview and self.preview_window.running:
            # self.video_manager.make_video(gif=False, mp4=True)
            self.preview_window.destroy()

            # clean the frames
            # frames_dir = os.path.join(self.output_dir, "frames")
            # if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
            #     os.removedirs(frames_dir)
            # self.video_manager.clean_frames() # BUG

            with imageio.get_writer(os.path.join(self.output_dir, "rendered.mp4"), fps=frame_rate) as writer:
                for image in frame_images:
                    image = np.flip(np.transpose(image, (1, 0, 2)), axis=0)
                    image_uint8 = (image * 255).astype(np.uint8)
                    writer.append_data(image_uint8)

        log("sph solver run complated")