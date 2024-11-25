import argparse 
from utils import run_docker_with_cmd, create_common_parser

def run_test(args):
    test_cmd = "cd optical && python code/test_sintel.py \
                --checkpoint checkpoints/best.ckpt \
                --uncertainity"
    if args.dataset == "sintel":
        test_cmd += " --data_list code/dataset_paths/MPI_Sintel_Final_train_list.txt"
    if args.dataset == "flyingchairs2":
        test_cmd += " --data_list code/dataset_paths/flyingchairs2.txt"
    run_docker_with_cmd(test_cmd, args)

def main():
    parser = argparse.ArgumentParser(description='start docker instance')
    create_common_parser(parser)
    args = parser.parse_args()

    run_test(args)

if __name__ == '__main__':
    main()
