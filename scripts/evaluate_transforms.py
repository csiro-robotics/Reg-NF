"""
Code is to take transformation numpy file, determine the scene and object data 
folders associated with them, get the GT transformation from scene to object, 
then evaluate mean squared difference between x,y,z translation and RPY rotation
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Dict
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd

# TODO need to sort through all of these and see what should not be made public
OBJECT_FULLNAMES = {"fc": "fancy_chair", "fc-nop": "fancy_chair_no_pillow", "c": "chair", "dc": "dining_chair",
                    "matc": "matrix_chair", "tbl": "table", "etbl": "end_table", "wtbl": "willow_table"}
SCENE_FULLNAMES = {"fc-room": "fancy_chair_room",
                   "ac-room": "all_chair_room", "atbl-room": "all_table_room"}
# BELOW ARE ALL SUPER HACKY
DESCRIPTORS = ["red", "green", "nodepth", "short", "early", "vearly", "r200"]
SDF_ROOT = Path("outputs")
MODEL_DEFAULT = "neus-facto"
IMG_SIZE_DEFAULT = "512"
OBJ_RENDER_DEFAULT = "rtxi"
SCENE_DATA_ROOT = Path("data/omniverse/scenes")
OBJECT_DATA_ROOT = Path("data/omniverse/objects")
SCENE_OBJECT_PRIMS = {
    "chair": ["WorldDellwood_DiningChair", "WorldDutchtown_Chair"],
    "table": ["WorldAppleseed_CoffeeTable"]}
PRIM_TO_OBJECT_SHORT = {
    "Dellwood_DiningChair": "dc",
    "Dutchtown_Chair": "fc",
    "Waiting": "c",
    "Dutchtown_Chair_no_pillows": "fc-nop",
    "The_Matrix_Red_Chesterfield_Chair": "matc",
    "Appleseed_CoffeeTable": "tbl",
    "Willow": "wtbl",
    "Appleseed_EndTable": "etbl"
}

# TODO adapt below from permute render


def decompose_tmat_file(transform_filename: Path) -> Tuple[Path, Path, Path, str, Dict]:
    """
    Get the scene sdf, object sdf and data path that is referred to by a transformation file

    Args:
        transform_filename (Path): The transformation numpy file in question

    Returns:
        Tuple[Path, Path]: The paths to the object and scene meta_data.json files from the
        data used to train the neural field models used to calculate the transformation matrix ...
    """
    basename = transform_filename.name[:-4]  # remove .npy
    # NOTE currently convention is <scene_model>-2-<obj_model>_<scene_obj_prim>
    # remove scene object prims
    components, scene_obj_prim = basename.split("_", maxsplit=1)
    # TODO this is messy. Account for singleview parameter
    singleview = False
    if "singleview" in scene_obj_prim:
        singleview = True
        scene_obj_prim = scene_obj_prim.split("_", maxsplit=1)[1]

    scaleexp = False
    if "scale" in scene_obj_prim:
        scaleexp = True
        scene_obj_prim = scene_obj_prim.split("_", maxsplit=1)[1]

    # Separate all the components of the object and scene model setup
    components = components.split("-")
    sep_idx = components.index("2")

    # Extract scene model name and path
    scene_components = components[:sep_idx]
    scene_suffixes = [suff for suff in DESCRIPTORS if suff in scene_components]
    scene_suffix_idx = len(scene_components) if len(scene_suffixes) == 0 else \
        min([scene_components.index(suff) for suff in scene_suffixes])

    # Extract data path
    scene_short_name = "-".join(scene_components[:scene_suffix_idx])
    scene_data_name = "_".join([SCENE_FULLNAMES[scene_short_name],
                                IMG_SIZE_DEFAULT] +
                               [c for c in scene_components[scene_suffix_idx:]
                                if c != "nodepth" and c != "early" and c != "vearly"])

    # Extract object data path
    obj_components = components[sep_idx+1:]
    obj_suffixes = [suff for suff in DESCRIPTORS if suff in obj_components]
    obj_suffix_idx = len(obj_components) if len(obj_suffixes) == 0 else \
        min([obj_components.index(suff) for suff in obj_suffixes])
    object_short_name = "-".join(obj_components[:obj_suffix_idx])
    object_data_name = "_".join([OBJECT_FULLNAMES[object_short_name],
                                 IMG_SIZE_DEFAULT] +
                                [c for c in obj_components[obj_suffix_idx:]
                                 if c != "nodepth"] +
                                [OBJ_RENDER_DEFAULT])  # note rtxi is default rendering type for objects
    object_json = OBJECT_DATA_ROOT / object_data_name / "meta_data.json"

    scene_json = SCENE_DATA_ROOT / scene_data_name / "meta_data.json"

    # Figure out the scene's GT object prim (used to apply offset)
    scene_obj_data_name = "_".join([OBJECT_FULLNAMES[PRIM_TO_OBJECT_SHORT[scene_obj_prim[5:]]],
                                    IMG_SIZE_DEFAULT, OBJ_RENDER_DEFAULT])
    scene_obj_json = OBJECT_DATA_ROOT / scene_obj_data_name / "meta_data.json"

    experiment_summary_dict = {
        "scene_short_name": scene_short_name,
        "scene_suffixes": scene_suffixes,
        "object_short_name": object_short_name,
        "object_suffixes": obj_suffixes,
        "singleview": singleview,
        "scaleexp": scaleexp
    }

    # Remove World from primitive name
    return scene_json, object_json, scene_obj_json, scene_obj_prim[5:], experiment_summary_dict

####################### MAIN FUNCTION ##################################


# TODO fix naming conventions!
parser = ArgumentParser()
parser.add_argument("--tmat_file", "-t", type=str,
                    help="transformation numpy file to go from some scene to some object")
parser.add_argument("--result_csv", "-r", type=str,
                    help="Location of csv file to append results into")
parser.add_argument("--run_id", "-i", help="id number of run for this" +
                    "experiment setting (used when multiple runs under " +
                    "the same conditions exist). Default 1", type=int, default=1)
args = parser.parse_args()

# Extract the predicted transformation matrix
tmat_path = Path(args.tmat_file)
pred_tmat = np.load(tmat_path)

# Determine the location of the data metafiles for both scene and object
# Do this based on transformation matrix naming convention
scene_meta, object_meta, scene_obj_meta, prim_name, experiment_summary = decompose_tmat_file(
    tmat_path)

# Load pose of object within scene
# TODO make into function not copy pastes
with open(scene_meta, "r") as f:
    scene_data = json.load(f)

# go through all instances in the world and find the one with the correct object primitive
# NOTE currently assumes object primitive is under "World/<prim_name>"
sobj_to_sworld = None
for inst_id, inst_info in scene_data["instances"].items():
    if inst_info["prim_path"] == f"/World/{prim_name}":
        sobj_to_sworld = np.array(inst_info["world_pose"])
        break
if sobj_to_sworld is None:
    raise ValueError(
        f"primitive '/World/{prim_name}' not found in {scene_meta}")

# Get the gt to world transform for the object world
with open(object_meta, "r") as f:
    object_data = json.load(f)

# Get the scaling factor for the object world of interest
# NOTE object world origin is the same as object NeRF origin
# it is also assumed that object and scene gt (not normalised)
# worlds are at the same scale for all objects
oworld_to_gt = np.array(object_data["worldtogt"])
gt_to_oworld = np.linalg.inv(oworld_to_gt)


# Get the offset for the scene object from it's base to it's centre
# This pose should be the same as the object's NeRF pose (and should not be scaled)
# NOTE this is the approximate object centre not the absolute centre
# but works out for the nerf2nerf and sdf2sdf analysis
with open(scene_obj_meta, "r") as f:
    # This assumes object file will only ever have one object
    # this is in GT scale
    gt_sobj_base_to_origin = json.load(f)["instances"]["1"]["gt_pose"]

object_obj_pose = np.array(object_data["instances"]["1"]["world_pose"])

# Calculate the gt tmat of scene to object
gt_tmat = gt_to_oworld @ gt_sobj_base_to_origin @ np.linalg.inv(sobj_to_sworld)

# Calculate evaluation as mse for translation, rotation and scale components seperately.
# Handy reference to remember
# https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
translation_rmse = np.sqrt(((gt_tmat[:3, 3] - pred_tmat[:3, 3]) ** 2).mean())
translation_rmse_cm = np.sqrt(
    (((oworld_to_gt @ gt_tmat)[:3, 3] -
      (oworld_to_gt @ pred_tmat)[:3, 3]) ** 2).mean())
gt_scale_factors = np.linalg.norm(gt_tmat[:3, :3], axis=0).reshape(1, -1)
pred_scale_factors = np.linalg.norm(pred_tmat[:3, :3], axis=0).reshape(1, -1)
scale_rmse = np.sqrt(((gt_scale_factors - pred_scale_factors)**2).mean())

# Decompose rotation matrix using scipy library
# NOTE to get rotation we must divide columns by the scaling factor for each axis
gt_rpy = R.from_matrix(
    gt_tmat[:3, :3] / gt_scale_factors).as_euler('xyz', degrees=False)
pred_rpy = R.from_matrix(
    pred_tmat[:3, :3] / pred_scale_factors).as_euler('xyz', degrees=False)

rotation_rmse = np.sqrt(((gt_rpy - pred_rpy)**2).mean())

print("################################################")
print(f"Tranformation File: {tmat_path.name} \nRun {args.run_id} Results:")
print(f"Translation RMSE (obj world): {translation_rmse:.4f}")
print(f"Translation RMSE (cm): {translation_rmse_cm:.4f}")
print(f"Rotation RMSE (radians):    {rotation_rmse:.4f}")
print(f"Scale RMSE: {scale_rmse:.4f}")
print("################################################")

# Save results to file
new_result = {
    # TODO confirm naming convention for objects and scene names in table
    "tmat_file": [tmat_path.name],
    "run_id": [args.run_id],
    "tranlsation_rmse": [translation_rmse],
    "translation_rmse_cm": [translation_rmse_cm],
    "rotation_rmse": [rotation_rmse],
    "scale_rmse": [scale_rmse],
    "true_scale": [gt_scale_factors[0][0]],
    "pred_scale": [pred_scale_factors[0][0]],
    "scene": [experiment_summary["scene_short_name"]],
    "scene_eval_object": [PRIM_TO_OBJECT_SHORT[prim_name]],
    "scene_depth": [False] if "nodepth" in experiment_summary["scene_suffixes"] else [True],
    "scene_early": [True] if "early" in experiment_summary["scene_suffixes"] else [False],
    "scene_vearly": [True] if "vearly" in experiment_summary["scene_suffixes"] else [False],
    "short_trajectory": [True] if "short" in experiment_summary["scene_suffixes"] else [False],
    "object": [experiment_summary["object_short_name"]],
    "object_depth": [False] if "nodepth" in experiment_summary["object_suffixes"] else [True],
    "object_coloured": [True] if ("red" in experiment_summary["object_suffixes"]
                                  or "green" in experiment_summary["object_suffixes"]) else [False],
    "singleview": [experiment_summary["singleview"]]
}
new_df = pd.DataFrame(new_result)
save_file = Path(args.result_csv)

save_file.parent.mkdir(parents=True, exist_ok=True)

# Write results to file
new_df.to_csv(save_file, index=False,
              mode="a" if save_file.exists() else "w",  # Append if file exists already
              header=not save_file.exists())  # Only write headers if doesn't exist
