import Fluid
import argparse

def main():
    parser = argparse.ArgumentParser(description="fluid simulation")

    parser.add_argument("--method", type=str, default="sph", help="fluid simulation algorithm")
    parser.add_argument("--length", type=float, default=0.3, help="simulation length in seconds")
    parser.add_argument("--scene", type=str, default="scenes/example.json", help="scene config file path")
    parser.add_argument("--enable_output", action=argparse.BooleanOptionalAction, help="save simulation to disk")
    parser.add_argument("--output", type=str, default="output/particles", help="output path for simulation result")
    parser.add_argument("--enable_preview", action=argparse.BooleanOptionalAction, help="open a preview window")

    args = parser.parse_args()

    Fluid.simulation_entry(args=args)

if __name__ == "__main__":
    main()