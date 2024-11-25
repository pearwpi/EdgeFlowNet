import argparse 
from utils import run_docker_with_cmd, create_common_parser

def run_test(args):
    test_cmd = f"cd optical && python code/train.py --Dataset {args.dataset} --data_list code/dataset_paths/"
    run_docker_with_cmd(test_cmd, args)

def main():
    parser = argparse.ArgumentParser(description='start docker instance')
    create_common_parser(parser)
    args = parser.parse_args()

    run_test(args)

if __name__ == '__main__':
    main()
