from argparse import ArgumentParser
from pathlib import Path


def write_command(save_file: str, np_file: Path, result_file: str, run_id: int = 1):
    """
    Write command to bash script for evaluation command on a numpy transformation file

    Args:
        save_file (str): bash file to save commands to
        np_file (Path): Numpy file to run evaluation on
        result_file (str): result .csv file to save evaluation results to
        run_id (int, optional): run id for the experiment 
        (used for multiple runs under same conditions). Defaults to 1.
    """
    # check if file is numpy file and if not return
    if np_file.suffix != ".npy":
        return
    with open(save_file, "a") as f:
        f.write(
            f'echo "{np_file.name} - {run_id}"\n')
        f.write(
            f"python scripts/evaluate_transforms.py --tmat_file {np_file} " +
            f"--result_csv {result_file} --run_id {run_id}")
        f.write("\n")


# MAIN FUNCTION!
parser = ArgumentParser()
parser.add_argument("--transforms_folder", "-t",
                    type=str, default="transforms")
parser.add_argument("--results_file", "-r", type=str,
                    default="eval_results.csv")
parser.add_argument("--bash_file", "-b", type=str, default="permute_eval.sh")
parser.add_argument("--multiple_runs", action="store_true")
args = parser.parse_args()

filename = args.bash_file
# Create new file for filename (note this will overwrite any file with existing name)
with open(filename, "w") as f:
    f.write("#!/bin/sh\n")

# Go through contents of transforms folder and use numpy files to determine combo
root_folder = Path(args.transforms_folder)
for content in root_folder.iterdir():
    if args.multiple_runs and content.is_dir and content.name.isdigit():
        for sub_content in content.iterdir():
            write_command(filename, sub_content,
                          args.results_file, int(content.name))
    else:
        write_command(filename, content, args.results_file, 1)
