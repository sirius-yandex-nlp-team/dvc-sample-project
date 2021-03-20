import yaml
import json
import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def main():
    grid = yaml.load(open("grid.yaml"))
    grid = list(product_dict(**grid))
    grid_json = json.dumps([json.dumps(i) for i in grid])
    print(f"::set-output name=matrix::{grid_json}")

if __name__ == "__main__":
    main()
