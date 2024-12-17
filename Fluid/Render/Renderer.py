import os, json
import cv2
from tqdm import tqdm
import mitsuba as mi

from Fluid._basic import *

class Renderer:
    def __init__(self, args):
        self.cmd_args = args
        self.scene_cfg = dict()
        self.frame_data_mitsuba_dict = dict()
        self.rendered_images_path = list()

    def __del__(self):
        ...

    def build_common_frame(self):
        # common part
        # self.frame_data_mitsuba_dict = mi.cornell_box() # for test

        # calc realpath of resource file, texture etc
        scene_file_path = os.path.dirname(os.path.abspath(self.cmd_args.scene))

        render_cfg = self.scene_cfg["render"]
        width, height = render_cfg["width"], render_cfg["height"]
        env_texture = os.path.join(scene_file_path, render_cfg["env_texture"])
        camera_location = render_cfg["camera_location"]
        camera_target = render_cfg["camera_target"]

        obj_glass_bsdf = mi.load_dict({
            "type": "conductor",
            "material": "Au"
        })
        base_plane_bsdf = mi.load_dict({
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "to_uv": mi.ScalarTransform4f().scale([20, 20, 1])
            }
        })
        wall_bsdf = mi.load_dict({
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": 1
            },
            "int_ior": 1.9
        })
        self.frame_data_mitsuba_dict = {
            "type": "scene",
            "integrator": {
                "type": "path",
            },
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "near_clip": 0.1,
                "far_clip": 1000.0,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=mi.Point3f(*camera_location),
                    target=mi.Point3f(*camera_target),
                    up=mi.Point3f(*world_up)
                ),
                "film": {
                    "type": "hdrfilm",
                    "rfilter": {
                        "type": "box"
                    },
                    "width": width,
                    "height": height,
                },
                "sampler": {
                    "type": "independent",
                    "sample_count": 4,
                },
            },
            "env_emitter": {
                "type": "envmap",
                "filename": env_texture
            },
            "base_plane": {
                "type": "rectangle",
                "material": base_plane_bsdf,
                "to_world": mi.ScalarTransform4f().rotate([1,0,0], -90).scale([5,5,1])
            },
        }

        # build walls
        parameters = self.scene_cfg["parameters"]
        domain_start, domain_end = parameters["domain_start"], parameters["domain_end"]
        # local transform?
        self.frame_data_mitsuba_dict["wall_xy_1"] = {
            "type": "rectangle",
            "material": wall_bsdf,
            "to_world": mi.ScalarTransform4f().translate([
                (domain_end[0] + domain_start[0]) * 0.5,
                (domain_end[1] + domain_start[1]) * 0.5,
                domain_start[2]
            ]).scale([
                (domain_end[0] - domain_start[0]) * 0.5,
                (domain_end[1] - domain_start[1]) * 0.5,
                1
            ])
        }
        self.frame_data_mitsuba_dict["wall_yz_1"] = {
            "type": "rectangle",
            "material": wall_bsdf,
            "to_world": mi.ScalarTransform4f().translate([
                domain_start[0],
                (domain_end[1] + domain_start[1]) * 0.5,
                (domain_end[2] + domain_start[2]) * 0.5
            ]).rotate(
                [0,1,0], 90
            ).scale([
                (domain_end[2] - domain_start[2]) * 0.5,
                (domain_end[1] - domain_start[1]) * 0.5,
                1
            ])
        }
        self.frame_data_mitsuba_dict["wall_yz_2"] = {
            "type": "rectangle",
            "material": wall_bsdf,
            "to_world": mi.ScalarTransform4f().translate([
                domain_end[0],
                (domain_end[1] + domain_start[1]) * 0.5,
                (domain_end[2] + domain_start[2]) * 0.5
            ]).rotate(
                [0,1,0], -90
            ).scale([
                (domain_end[2] - domain_start[2]) * 0.5,
                (domain_end[1] - domain_start[1]) * 0.5,
                1
            ])
        }
        
        rigid_bodies = self.scene_cfg["rigid_bodies"]
        for rigid_body in rigid_bodies:
            id = rigid_body["id"]
            obj_file = os.path.join(scene_file_path, rigid_body["obj_file"])
            offset = rigid_body["offset"]
            self.frame_data_mitsuba_dict[f"rigid_bodies_{id}"] = {
                "type": "obj",
                "material": obj_glass_bsdf,
                "filename": obj_file,
                "to_world": mi.Transform4f().translate(offset)
            }

    def build_frame(self, input_json_file: str):
        self.build_common_frame()

        parameters = self.scene_cfg["parameters"]
        particle_mass, density = parameters["particle_mass"], parameters["density"]
        particle_radius = calc_radius(particle_mass, density)

        single_particle_group = mi.load_dict({
            "type": "shapegroup",
            "base": {
                "type": "sphere",
                "radius": particle_radius,
                "bsdf": {
                    "type": "diffuse"
                }
            }
        })
        frame_data = None
        if self.cmd_args.format == "particles":
            with open(input_json_file, "r") as file:
                frame_data = json.load(file) # particles part
                particles = frame_data["particles"]
                particle_idx = 0
                for particle in particles:
                    particle_idx += 1
                    self.frame_data_mitsuba_dict[f"particles_instance_{particle_idx}"] = {
                        "type": "instance",
                        "shapegroup": single_particle_group,
                        "to_world": mi.Transform4f().translate(particle)
                    }

        elif self.cmd_args.format == "surface":
            log("render part for surface mesh is under construction...")

    def render_frame(self, output_image_file: str):
        result = mi.render(mi.load_dict(self.frame_data_mitsuba_dict), spp=self.cmd_args.spp)
        mi.util.write_bitmap(output_image_file, result)

    def encode_video(self):
        if len(self.rendered_images_path) < 1:
            log("there is nothing to encode to video, exiting...")
            return

        output_dir = self.cmd_args.output
        output_video = os.path.join(output_dir, "render_res.avi")
        fps = self.scene_cfg["parameters"]["frame_rate"]

        render_cfg = self.scene_cfg["render"]
        width, height = render_cfg["width"], render_cfg["height"]

        video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))

        for image in self.rendered_images_path:
            video.write(cv2.imread(image))

        video.release()
        log("video encode complated")

    @log_time
    def render_all(self):
        self.scene_cfg = load_scene(self.cmd_args.scene)

        if self.scene_cfg is None:
            log("starting render failed, because the scene file is missing")
            return
        
        input_dir = self.cmd_args.input
        output_dir = self.cmd_args.output
        if not os.path.isdir(input_dir):
            log("please check your data input path, it is not readable")
            return

        os.makedirs(output_dir, exist_ok=True)

        data_files = os.listdir(input_dir)
        data_files = [i for i in data_files if i.endswith(".json")]
        data_files = sorted(data_files)
        
        log(f"total render frames count is {len(data_files)}")

        remove_json_extension = slice(None, -5)
        self.rendered_images_path.clear()

        enter_bar()
        for data_file in tqdm(data_files, desc="rendering frames"):
            input_json_file = os.path.join(input_dir, data_file)
            output_image_file = os.path.join(output_dir, f"{data_file[remove_json_extension]}.png")
            self.build_frame(input_json_file)
            self.render_frame(output_image_file)
            self.rendered_images_path.append(output_image_file)
        exit_bar()

        log("render task complated")
        if self.cmd_args.encode_video is True:
            log("starting encode to video")
            self.encode_video()
