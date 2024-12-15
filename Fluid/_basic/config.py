import os, json
import Fluid._basic.message as m

def load_scene(scene_file_path: str) -> dict:
    cfg = dict()
    try:
        with open(scene_file_path, "r") as file:
            cfg = json.load(file)
    except FileNotFoundError:
        m.log(f"{scene_file_path} is not a available scene file")
        return None
    except json.JSONDecodeError:
        m.log(f"{scene_file_path} is not a json file")
        return None
    except Exception as error:
        m.log(f"{scene_file_path} load failed:", str(error))
        return None

    return cfg