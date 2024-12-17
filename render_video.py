import Fluid
import argparse

def main():
    parser = argparse.ArgumentParser(description="render simulation result")

    parser.add_argument("--format", type=str, default="particles", help="particles or surface")
    parser.add_argument("--spp", type=int, default=1, help="sample per pixel")
    parser.add_argument("--scene", type=str, default="scenes/example.json", help="scene config file path")
    parser.add_argument("--input", type=str, default="output/particles", help="simulation result path, render data")
    parser.add_argument("--output", type=str, default="output/render_result", help="output path for render result")
    parser.add_argument("--encode_video", type=bool, default=False, help="encode the images to video")

    args = parser.parse_args()

    Fluid.render_entry(args=args)

if __name__ == "__main__":
    main()