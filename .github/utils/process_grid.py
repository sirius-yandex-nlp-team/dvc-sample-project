import yaml
import json

def main():
    grid = yaml.load(open("grid.yaml"))
    grid_json = json.dumps(grid)
    print)f"::set-output name=matrix::{grid_json}")

if __name__ == "__main__":
    main()
