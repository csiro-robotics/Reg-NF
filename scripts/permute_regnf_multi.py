import numpy as np
import os.path as osp
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict
import yaml
import copy

IMAGE_SIZES = [512]
MODEL_TYPES = ["neus-facto"]
CLASS_NAMES = ['chair', "table"]
OBJECT_ABBREVIATIONS = {"fancy_chair": "fc", "fancy_chair_no_pillow": "fc-nop",
                        "matrix_chair": "matc", "table": "tbl", "end_table": "etbl", "willow_table": "wtbl",
                        "chair": "c", "dining_chair": "dc"}
SCENE_ABBREVIATIONS = {"fancy_chair_room": "fc-room",
                       "all_chair_room": "ac-room", "all_table_room": "atbl-room"}
OBJ_SCALES = ["0.1", "0.25", "0.5", "2", "scene"]

# Create model name from traits provided
DEFAULT_PARAMETERS = {"img_size": IMAGE_SIZES[0],
                      "model": MODEL_TYPES[0],
                      "short_trajectory": False,
                      "early": False,
                      "vearly": False,
                      "oscale": None}


def get_config_path(
        model_name: str,
        model_str: str,
        obj: bool,
        early: bool = False,
        vearly: bool = False) -> Path:
    """
    Extract the configuration file path for the most recent version of a 
    given model defined by the model string. Requires model to have at least
    one saved checkpoint file.

    Args:
        model_name (str): type of model being used (e.g. neus-facto)
        model_str (str): string descriptor for the model type. Usually "experiment name" in sdfstudio.
        obj (bool): flag depicting if config is that of an object or not. If not assume scene
        early (bool): flag for if we want an undertrained model (look for suffix early in filename)
        vearly (bool): flag for if we want a very undertrained model (look for suffix vearly in filename)

    Raises:
        RuntimeError: Raises error if no model file can be found

    Returns:
        Path: pathlib Path for the config.yaml file for the most recent version of the model
    """
    type_str = "objects" if obj else "scenes"
    models_root = Path("outputs") / "omniverse" / \
        type_str / model_name / model_str
    # Go through most recent folders until you find one that has a models folder
    # NOTE not certain this is the "best" way to get the most recent experiment results
    for model_option in reversed(sorted(models_root.iterdir())):
        for content in model_option.iterdir():
            if content.name == "sdfstudio_models":
                if early:
                    config_name = model_option / "config_early.yml"
                    # Make early config file if doesn't exist
                    if not config_name.exists():
                        raise ValueError(
                            f"{config_name} does not exist!! Please make one")
                    return config_name
                elif vearly:
                    config_name = model_option / "config_vearly.yml"
                    # Make early config file if doesn't exist
                    if not config_name.exists():
                        raise ValueError(
                            f"{config_name} does not exist!! Please make one")
                    return config_name

                return model_option / "config.yml"

    raise RuntimeError(
        f"No valid config with sdfstudio_models was found in {models_root}")


def get_object_model_name(
        object_name: str,
        parameters: Dict):
    """
    Get the model name (experiment name) for an object

    Args:
        scene_name (str): The long version of the object name (not abbreviated)
        parameters (dict): Any parameters that highlight difference in the model

    Returns:
        str: scene's model name
    """
    model_str = parameters["model"]
    obj_str = f"-{OBJECT_ABBREVIATIONS[object_name]}"
    img_size_str = f"-{parameters['img_size']}"
    scale_str = "" if parameters["oscale"] is None else f"-scale{parameters['oscale']}"
    return model_str + obj_str + img_size_str + scale_str


def get_scene_model_name(
        scene_name: str,
        parameters: Dict) -> str:
    """
    Get the model name (experiment name) for a scene

    Args:
        scene_name (str): The long version of the scene name (not abbreviated)
        parameters (dict): Any parameters that highlight difference in the model 

    Returns:
        str: scene's model name
    """
    model_str = parameters["model"]
    scene_str = f"-{SCENE_ABBREVIATIONS[scene_name]}"
    traj_len_str = "-short" if parameters["short_trajectory"] else ""
    img_size_str = f"-{parameters['img_size']}"
    return model_str + scene_str + img_size_str + traj_len_str


def min_model_name(
        model_name: str,
        default_vals: List[str]) -> str:
    """
    Extract the minimal model name, removing any parts
    of model name that match default parameters

    Args:
        model_name (str): original model name to be minimized
        default_vals (List[str]): string of default parameters to be ignored

    Returns:
        str: minimzed version of model name (minus default parameters)
    """
    min_name = model_name
    for default in default_vals:
        min_name = min_name.replace(f"{default}-", "")
        min_name = min_name.replace(f"-{default}", "")
    return min_name


def get_transform_path(
        root_path: str,
        object_model_name: str,
        scene_model_name: str,
        min_name: bool = False,
        early: bool = False,
        vearly: bool = False) -> Path:
    """
    Define a transformation file path for a particular object/scene pairing

    Args:
        root_path (str): root location to store transforms within
        object_model_name (str): Name of the object model (experiment name)
        scene_model_name (str): Name of the scene model (experiment name)
        min_name (bool, optional): Flag for if only non-default parameters should be listed. Defaults to False.

    Returns:
        Path: Transformation numpy file (.npy) to save transformations to
    """
    # TODO check if absolute path required
    transform_path = Path(root_path)
    early_str = "-early" if early else "-vearly" if vearly else ""
    if min_name:
        # Only include parameters in name that are not default
        default_vals = [DEFAULT_PARAMETERS["img_size"],
                        DEFAULT_PARAMETERS["model"]]
        filename = f"{min_model_name(scene_model_name, default_vals)}{early_str}-2-{min_model_name(object_model_name, default_vals)}.npy"
    else:
        filename = f"{scene_model_name}{early_str}-2-{object_model_name}.npy"
    return transform_path / filename


def get_data_path(
        scene_name: str,
        parameters: Dict) -> Path:
    """
    Find the data path given a scene and a set of parameters

    Args:
        scene_name (str): Scene name in long form
        parameters (Dict): Parameters dictionary for scene data

    Returns:
        Path: Data path (from sdfstudio_NFDT root) to the 
    """
    data_path = Path("data") / "omniverse" / "scenes"
    traj_str = "_short" if parameters["short_trajectory"] else ""
    return data_path / f"{scene_name}_{parameters['img_size']}{traj_str}"


def make_run_cmd(
        scene_config: Path,
        object_config: Path,
        transform_path: Path,
        data_path: Path,
        multiple: bool = False) -> str:
    """
    Compose the command line argument for sdf2sdf


    Args:
        scene_config (Path): Path to scene config.yml file
        object_config (Path): Path to object config.yml file
        transform_path (Path): Path to desired transformation.npy file
        data_path (Path): Path to scene data

    Returns:
        str: Command line argument for sdf2sdf_multiview
    """
    idx_str = '${IDX}'if multiple else ""
    baseline = f"python scripts/regnf.py --use_visdom {int(args.visdom)} --use_open3d {int(args.open3d)}"
    cmd = baseline + f" --scene_sdf_path {scene_config}"
    cmd += f" --object_sdf_path {object_config}"
    cmd += f" --transform_savepath {transform_path.parent / idx_str / transform_path.name}"
    if args.open3d:
        cmd += f" --vis_output {Path(args.open3d_output) / idx_str / transform_path.name[:-4]}"
    if args.save_init:
        cmd += f" --save_init 1"
    if args.save_fgr:
        cmd += f" --save_fgr 1"
    if not args.all_scene_objects:
        cmd += " --all_scene_objects 0"
    cmd += f" sdfstudio-data --data {data_path}"

    return cmd


def main():
    object_names = args.object_names
    scene_names = args.scene_names
    # TODO fix this. It's hacky even by this code base standards
    object_test_params = {
        "oscale": [None] if not args.obj_scales else OBJ_SCALES}
    scene_test_params = {
        "short_trajectory": args.scene_short, "early": args.scene_early, "vearly": args.scene_vearly}
    filename = args.bash_file
    # Create new file for filename (note this will overwrite any file with existing name)
    with open(filename, "w") as f:
        f.write("#!/bin/sh\n")
        if args.num_tests > 0:
            f.write("for IDX in ")
            for i in range(args.num_tests):
                f.write(f"{i+1:02} ")
            f.write("\ndo\n")
            f.write('echo "STARTING ITERATION ${IDX}"\n')
    # Go through all scenes and scene parameters
    for scene in scene_names:
        for short_traj in scene_test_params["short_trajectory"]:

            for s_early in scene_test_params["early"]:
                for s_vearly in scene_test_params["vearly"]:
                    if bool(s_early) and bool(s_vearly):
                        continue
                    # Set up parameters for config path creation
                    scene_model_params = copy.deepcopy(DEFAULT_PARAMETERS)
                    scene_model_params["short_trajectory"] = bool(
                        short_traj)
                    scene_model_params["early"] = bool(s_early)
                    scene_model_params["vearly"] = bool(s_vearly)
                    scene_model_name = get_scene_model_name(
                        scene, scene_model_params)
                    data_path = get_data_path(scene, scene_model_params)
                    # Go through all objects and object parameters
                    for object in object_names:
                        # TODO the following is hacky even for this code base
                        for o_scale in object_test_params["oscale"]:
                            # Set up parameters for config path creation
                            obj_model_params = copy.deepcopy(
                                DEFAULT_PARAMETERS)
                            obj_model_params["oscale"] = o_scale
                            obj_model_name = get_object_model_name(
                                object, obj_model_params)
                            transform_path = get_transform_path(
                                args.transform_output,
                                obj_model_name, scene_model_name, True,
                                scene_model_params["early"], scene_model_params["vearly"])
                            # Append new command to file
                            with open(filename, "a") as f:
                                f.write(
                                    f"echo {osp.basename(transform_path)}\n")
                                f.write(make_run_cmd(
                                    get_config_path(
                                        scene_model_name,
                                        scene_model_params["model"],
                                        False,
                                        scene_model_params["early"],
                                        scene_model_params["vearly"]),
                                    get_config_path(
                                        obj_model_name, obj_model_params["model"], True),
                                    transform_path,
                                    data_path,
                                    args.num_tests > 0
                                ))
                                f.write("\n")
                            print(
                                f"Created command for {osp.basename(transform_path)}")
    if args.num_tests > 0:
        with open(filename, "a") as f:
            f.write("done\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene_names", "-s", nargs="+",
                        type=str, choices=list(SCENE_ABBREVIATIONS.keys()),
                        help="names of scenes to permute over (note that short trajectory" +
                        "is a different option to the scene itself)")
    parser.add_argument("--object_names", "-o", nargs="+",
                        type=str, choices=list(OBJECT_ABBREVIATIONS.keys()),
                        help="names of the base objects to permute over")
    parser.add_argument("--visdom", action="store_true",
                        help="whether to include visdom visualisation in commands")
    parser.add_argument("--open3d", action="store_true",
                        help="whether to include open3d visualisation in commands")
    parser.add_argument("--save_init", action="store_true",
                        help="whether to include saving init transform in commands")
    parser.add_argument("--save_fgr", action="store_true",
                        help="whether to include saving fgr transform in commands")
    parser.add_argument("--transform_output", type=str, default="transforms",
                        help="root location to store transforms")
    parser.add_argument("--open3d_output", type=str, default="open3d_imgs")
    parser.add_argument("--bash_file", "-b", default="permute_regnf.sh",
                        help="name of bash file to save python commands to")
    parser.add_argument("--scene_short", "-ss", type=int,
                        nargs="+", default=[0], choices=[0, 1],
                        help="what options for scene trajectories should be permuted over." +
                        " (0) is standard long trajectory and (1) is short trajectory")
    parser.add_argument("--scene_early", "-se", type=int,
                        nargs="+", default=[0], choices=[0, 1],
                        help="what options for having and early scene model should be permuted over." +
                        "(1) for models that use earliest checkpoint data and (0) for those that don't" +
                        " NOTE cannot run vearly and early as both true")
    parser.add_argument("--scene_vearly", "-sve", type=int,
                        nargs="+", default=[0], choices=[0, 1],
                        help="what options for having and early scene model should be permuted over." +
                        "(1) for models that use very early checkpoint data and (0) for those that don't." +
                        " NOTE cannot run vearly and early as both true")
    parser.add_argument("--all_scene_objects", action="store_true")
    parser.add_argument("--obj_scales", help="should we test all non-standard scales",
                        action="store_true")
    parser.add_argument("--num_tests", type=int, default=-1)
    args = parser.parse_args()

    main()
