from __future__ import annotations

# Standard imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import copy
import os.path as osp
import os
import json
import torch
import numpy as np
from PIL import Image
import tyro

# nerfstudio imports
# TODO understand why this import is needed to stop circular import
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.utils.eval_utils import eval_setup
from scripts.render_mesh import _interpolate_trajectory

# New render imports
from render_sdf_utils import render_scene_img, transform_points, get_model_gt_meta, new_obj_s2o_tmat
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_HSV2RGB

COLOUR_TO_HUE = {'red': 0, 'yellow': 30, 'green': 60,
                 'cyan': 90, 'blue': 120, 'magenta': 150}


@dataclass
class RenderFusedSDFs:
    """
    This code is largely copied from the render_mesh.py script
    Aim is to step through a series of camera poses based on a captured scene and render that scene model
    with overlays or replacements done for any learnt object models
    """

    # Path to the main scene sdf YAML
    scene_sdf_path: Path

    # Path to object transforms for the scene NPY format
    # NOTE assumes same order as object_sdf_paths
    object_transform_paths: List[Path] = field(default_factory=list)

    # Path to object sdf YAMLs (should match objects in object_transform_paths)
    object_sdf_paths: List[Path] = field(default_factory=list)

    # What colours to render objects as
    object_colours: List[str] = field(default_factory=list)

    # Path to object sdfs that will replace those originally used in the transform (in order of object transform paths)
    replace_sdf_paths: List[Path] = field(default_factory=list)

    # List of instance IDs to be removed from scene in "replace" mode.
    # NOTE assumes same order as object_sdf_paths and any extra ids
    # will simply be removed from the scene (no replacement)
    scene_remove_ids: List[int] = field(default_factory=list)

    # Name of the output file.
    output_path: Path = Path("renders/")

    # render mode to use for rendering. Can be replace or blend
    render_mode: str = "replace"

    # Sampler to be used in rendering. Options: original or uniform.
    # Note, original is typically neus-facto sampler which is quicker but can lead to
    # unusual visualisations after editing.
    sampler: str = "original"

    # data to use for getting camera trajectories (should be from scene SDF training)
    data: AnnotatedDataParserUnion = SDFStudioDataParserConfig()

    # Number of views to render (can be larger than number of images in data)
    # if -ve use the same number as were there originally
    num_views: int = 300

    def main(self) -> None:
        """Main function."""

        # Get the camera trajectories to be used for rendering from data (from render_mesh.py)
        outputs = self.data.setup()._generate_dataparser_outputs(
        )  # pylint: disable=protected-access
        if self.num_views > 0:
            camera_path = _interpolate_trajectory(
                cameras=outputs.cameras, num_views=self.num_views)
        else:
            camera_path = outputs.cameras

        # Load the models
        self.scene_pipeline = eval_setup(
            self.scene_sdf_path, test_mode='inference')[1]
        self.obj_pipelines = [eval_setup(object_yaml, test_mode='inference')[
            1] for object_yaml in self.object_sdf_paths]
        object_transforms = [
            torch.tensor(np.load(transform_file).astype(np.float32)).to(
                self.scene_pipeline.device)
            for transform_file in self.object_transform_paths
        ]

        # Replace object sdfs and corresponding transforms if desired
        if len(self.replace_sdf_paths) > 0:
            for oidx, new_config in enumerate(self.replace_sdf_paths):
                new_obj_pipeline = eval_setup(
                    new_config, test_mode='inference')[1]
                # Extract gt metadata for new and old obejct models
                # Note this comes from dataset used to train them
                new_meta = get_model_gt_meta(new_obj_pipeline)
                old_meta = get_model_gt_meta(self.obj_pipelines[oidx])

                old_tmat = object_transforms[oidx]
                # Calculate transformation from old to new tmat
                # with bases located in the same spot
                new_tmat = new_obj_s2o_tmat(old_tmat, old_meta, new_meta)

                # replace object pipeline with new object pipeline
                self.obj_pipelines[oidx] = new_obj_pipeline

                # replace old object transform with new object transform
                object_transforms[oidx] = new_tmat
                # From here the rest should be the same.

        # Determine what regions should be removed and what should be replaced
        # based upon the input parameters
        scene_remove_boxes = []
        object_replace_boxes = []
        if self.render_mode == "replace":
            # Define what objects should be removed from the scene model
            if len(self.scene_remove_ids) > 0:
                with open(self.data.data / "meta_data.json", "r") as f:
                    scene_metadata = json.load(f)
                scene_remove_boxes = [torch.tensor(scene_metadata["instances"][str(inst_id)]["box_corners"])
                                      for inst_id in self.scene_remove_ids]

            # Define what regions in scene should be replaced by object library models
            if len(object_transforms) > 0:
                for idx, pred_s2o in enumerate(object_transforms):
                    # Get the scene to object estimate
                    pred_o2s = torch.linalg.inv(pred_s2o)
                    # Get bounding box in object world
                    # TODO want better way of getting data path
                    datapath = Path(
                        self.obj_pipelines[idx].datamanager.config.dataparser.data)
                    with open(datapath / "meta_data.json", "r") as f:
                        o_meta = json.load(f)
                    oworld_box = torch.tensor(
                        o_meta["instances"][str(1)]["box_corners"]).to(pred_o2s.device)
                    sworld_box = transform_points(pred_o2s, oworld_box)
                    object_replace_boxes.append(sworld_box)

        object_models = [
            obj_pipeline.model for obj_pipeline in self.obj_pipelines]
        scene_model = self.scene_pipeline.model

        # go through all cameras in trajectory
        for cam_idx, scene_camera in enumerate(camera_path):
            # Generate scene level image
            # Create raybundles for scene
            scene_raybundle = scene_camera.generate_rays(
                0).to(self.scene_pipeline.device)

            # Original blended render
            # TODO replace with what is now in render multiple sdfs transforms.py
            if self.render_mode == "blend":
                with torch.no_grad():
                    scene_out = self.scene_pipeline.model.get_outputs_for_camera_ray_bundle(
                        scene_raybundle)
                scene_img = (scene_out["rgb"].cpu().numpy()
                             * 256).clip(0, 255).astype(np.uint8)
                new_img = copy.deepcopy(scene_img)
                obj_imgs = []
                obj_masks = []
                # go through all objects in list
                for obj_idx, obj_pipeline in enumerate(self.obj_pipelines):
                    # Create new camera with updated pose
                    obj_cam = copy.deepcopy(scene_camera)
                    # DO THE TRANSFORM
                    scene_to_object = object_transforms[obj_idx].to(
                        obj_cam.device)
                    obj_cam_to_worlds = torch.cat(
                        (obj_cam.camera_to_worlds, torch.tensor(
                            [[0.0, 0.0, 0.0, 1.0]]).to(device=obj_cam.device))
                    )

                    obj_cam.camera_to_worlds = (
                        scene_to_object @ obj_cam_to_worlds)[:-1, :]
                    # TODO check if should be transforming raybundle over camera
                    # If I can do that, I can remove the step of generating raybundles again
                    obj_raybundle = obj_cam.generate_rays(
                        0).to(obj_pipeline.device)
                    # TODO some sort of filtering based on object raybundle
                    # Don't need to render image for object not present
                    with torch.no_grad():
                        obj_out = obj_pipeline.model.get_outputs_for_camera_ray_bundle(
                            obj_raybundle)
                    obj_imgs.append(
                        (obj_out["rgb"].cpu().numpy() * 256).clip(0, 255).astype(np.uint8))

                    # NOTE we use accumulation here as a way of making an object mask.
                    # In object fields, accumulation is close to 1 everywhere there is an object
                    # Without this, we have "white space" all over our images
                    obj_masks.append(
                        np.squeeze(obj_out['accumulation'].cpu().numpy() > 0.75))

                    # Add colouration to the object in the image as desired
                    if obj_idx < len(self.object_colours):
                        hsv = cvtColor(obj_imgs[-1], COLOR_RGB2HSV)
                        hsv[..., 0] = COLOUR_TO_HUE[self.object_colours[obj_idx]]
                        hsv[..., 1] = 255
                        obj_imgs[-1] = cvtColor(hsv, COLOR_HSV2RGB)

                # Merge all images
                pixel_weightings = (
                    1/(np.sum([om for om in obj_masks], axis=0)+1))
                for obj_idx, obj_img in enumerate(obj_imgs):
                    mask = obj_masks[obj_idx]
                    scales = pixel_weightings[mask].reshape(-1, 1)

                    new_img[mask] = new_img[mask] * \
                        (1-scales) + obj_img[mask]*scales

            # Replacing rendering
            elif self.render_mode == "replace":
                new_img = render_scene_img(
                    scene_raybundle=scene_raybundle,
                    scene_model=scene_model,
                    object_models=object_models if len(
                        object_models) > 0 else None,
                    object_transforms=object_transforms if len(
                        object_models) > 0 else None,
                    scene_remove_boxes=scene_remove_boxes,
                    object_replace_boxes=object_replace_boxes,
                    object_colours=self.object_colours,
                    sampler=self.sampler
                )
            else:
                raise ValueError(
                    f"Invalid render_mode provided! {self.render_mode} is not replace or blend")
            # Save image
            img_name = osp.join(self.output_path, f"{cam_idx:06}.png")
            print(f"Saving {img_name}")
            Image.fromarray(new_img).save(img_name)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderFusedSDFs).main()


if __name__ == "__main__":
    entrypoint()
