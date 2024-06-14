#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
import os
import copy
import time
import string

import yaml
import tyro
import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
from torchtyping import TensorType
import open3d
import visdom

import robust_loss_pytorch.general
from scipy.spatial.transform import Rotation
from rich.console import Console

# TODO understand why this import is needed to stop circular import
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import

from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline
from scripts.render_mesh import _interpolate_trajectory

from scripts import regnf_utils as uu
from get_obj_cam_poses import get_obj_cam_poses

'''
Not a part of SDF Studio. File added by SH. Designed to run RegNF between two models produced by SDFStudio.
IMPORTANT: before running this code, start a visdom server by opening a terminal, loading the conda env then typing "visdom"
'''

# Constants that are used in the code base
CONSOLE = Console(width=120)
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
OBJECT_FULLNAMES = {"fc": "fancy_chair", "fc-nop": "fancy_chair_no_pillow", "c": "chair", "dc": "dining_chair",
                    "matc": "matrix_chair", "tbl": "table", "etbl": "end_table", "wtbl": "willow_table"}
ANALYSED_CLASSES = ["table", "chair"]
DESCRIPTORS = ["red", "green", "nodepth", "short", "early", "r200"]

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


@dataclass
class RegNF:
    # Path to the main scene sdf YAML
    scene_sdf_path: Path

    # Path to the library object sdf YAML
    object_sdf_path: Path

    # Name of the output transformation savefile.
    # NOTE assumes that "transforms" is in the path and
    # eval code assumes <scene_model>-2-<obj_model>.npy basename
    transform_savepath: Path = Path("transforms/fc-room-2-fc.npy")

    # Config file path
    config_path: Path = Path("configs/default_regnf.yaml")

    # Flag for if open3d visualisation should be provided
    use_open3d: int = 0

    # Path to save open3d visualisation outputs
    # Note: only applies when use_open3d is 1
    open3d_savepth: str = "open3d_imgs/"

    # Flag for if Visdom visualisations should be available
    use_visdom: int = 0

    # Flag for disabling object roll during optimization (good for locking object to ground plane)
    disable_roll: int = 0
    # Flag for disabling object pitch during optimization (good for locking object to ground plane)
    disable_pitch: int = 0

    # Flag for if object model should be evaluated against all labelled objects in the scene or just
    # object of the same underlying simulated model as provided (defined by prim_path in original dataset)
    all_scene_objects: int = 0

    # Flag for if we should save initialisation transforms before RegNF refinement
    save_init: int = 0

    # Flag for if we should save FGR transforms
    save_fgr: int = 0

    # Flag for if we should save transforms for each iteration of refinement
    save_iters: int = 0

    # Maximum number of iterations for optimizing object pose
    maxiters: int = 200

    # data to use for getting camera trajectories (should be from scene SDF training)
    data: AnnotatedDataParserUnion = SDFStudioDataParserConfig()

    def get_init_pose(self,
                      tmat: ArrayLike,
                      s_centroid: ArrayLike,
                      o_centroid: ArrayLike) -> ArrayLike:
        """
        Function for creating initial scene to library object pose estimate for RegNF from run_initial_pose_guess 
        output, ahead of iterative fine-tuning.
        TODO check is scene to object transform

        Args:
            tmat (ArrayLike): 4x4 Transformation matrix output by run_initial_pose_guess
            s_centroid (ArrayLike): 3, vector of estimated centroid for object within the scene
            o_centroid (ArrayLike): 3, vector of estimated centroid for object in object library

        Returns:
            ArrayLike: 4x4 Transformation matrix of the initial pose of scene to library object pose
        """

        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # this is the true and correct transformation for initializing RegNF
        full_init_transform = np.dot(tmat, trans_init)

        R = uu.make_rot_matrix(torch.tensor(0.0).to(device=self.scene_pipeline.device),
                               torch.tensor(0.0).to(
            device=self.scene_pipeline.device),
            torch.tensor(0.0).to(device=self.scene_pipeline.device))

        # move room object to (0,0,0)
        init_room_pose = np.zeros((4, 4))
        init_room_pose[:3, :3] = R.detach().cpu().numpy()
        init_room_pose[:, 3] = np.append(-s_centroid, 1)

        # move object model to (0,0,0)
        init_obj_pose = np.zeros((4, 4))
        init_obj_pose[:3, :3] = R.detach().cpu().numpy()
        init_obj_pose[:, 3] = np.append(-o_centroid, 1)

        # just translation and rotation
        pose_transform = np.linalg.inv(
            init_obj_pose) @ full_init_transform @ init_room_pose
        return pose_transform

    def setup_visualisations(self):
        """
        Set up the visualisations that can be used during RegNF
        """
        # start visdom
        vis = None
        vis_port = 8097
        if self.use_visdom:
            is_open = uu.check_socket_open('localhost', vis_port)
            retry = None
            while not is_open:
                retry = input(
                    f"visdom port ({vis_port}) not open, retry? (y/n) ")
                if retry not in ["y", "n"]:
                    continue
                if retry == "y":
                    is_open = uu.check_socket_open('localhost', vis_port)
                else:
                    break
            vis = visdom.Visdom(server='localhost', port=vis_port, env='Reg')

        # set up Open3D
        openvis = None
        if self.use_open3d:
            openvis = open3d.visualization.Visualizer()
            openvis.create_window(visible=True)

            # Make sure there is a folder to store new open3d images in
            Path(self.open3d_savepth).mkdir(parents=True, exist_ok=True)

        # Return Visdom and/or openvis Visualiser instance (None returned if)
        return vis, openvis

    def get_model_gt_meta(self, pipeline: Pipeline) -> Dict:
        """
        Extract the metadata for the provided model based on dataset used to train it.
        NOTE assumes data is available in same relative file structure as contained in model YAML

        Args:
            pipeline (Pipeline): Pipeline of model to attain metadata of

        Returns:
            Dict: Dictionary of metadata from data used to train provided model.
        """
        datapath = pipeline.datamanager.dataparser.config.data / \
            Path('meta_data.json')
        with open(datapath, 'r') as f:
            metadata = json.load(f)
        return metadata

    def setup_data(self) -> Tuple[Cameras, Cameras, List, Dict]:
        """
        Function for setting up all the data for the RegNF process.
        This includes:

            - loading models
            - Creating Cameras object for scene and library models
            - Collating list of all desired scene object metadata
            - Extracting library object metadata

        Returns:
            Tuple[Cameras, Cameras, List, Dict]: 
            Scene camera, library camera, 
            list of scene object metadata and library object metadata
        """
        # Load experiment config settings
        with open(self.config_path) as f:
            self.config_dict = yaml.safe_load(f)

        # load scene and object library models
        _, self.scene_pipeline, _ = eval_setup(
            self.scene_sdf_path, test_mode='inference')  # room
        _, self.obj_pipeline, _ = eval_setup(
            self.object_sdf_path, test_mode='inference')

        print("loaded all sdfstudio models")

        # Get camera parameters (assume they never change)
        outputs = self.data.setup()._generate_dataparser_outputs(
        )  # pylint: disable=protected-access
        camera_path = _interpolate_trajectory(
            cameras=outputs.cameras, num_views=1)
        first_scene_cam = camera_path[0]
        scene_cam = copy.deepcopy(first_scene_cam)
        obj_cam = copy.deepcopy(first_scene_cam)

        # Load scene metadata
        scene_metadata = self.get_model_gt_meta(self.scene_pipeline)

        # Extract instances present within the scene of desired analysis classes
        scene_insts = [inst for inst in scene_metadata['instances'].values()
                       if inst["class_name"] in ANALYSED_CLASSES]

        # TODO David to tidy this up more
        # HACK this is disgusting code This will only work if we have one object file
        # Assumes format is outpus/model_name/neus_facto/date/config.yml
        if not self.all_scene_objects:
            # Grab the names of the object models (e.g. fancy_chair)
            model_names = [
                Path(self.object_sdf_path).parents[2].name
            ]
            for idx, name in enumerate(model_names):
                # copied from evaluate
                # NOTE Assume that model_name starts with neus-facto
                obj_components = name.split("-")[2:]
                obj_suffixes = [
                    suff for suff in DESCRIPTORS if suff in obj_components]
                obj_suffix_idx = len(obj_components) if len(obj_suffixes) == 0 else \
                    min([obj_components.index(suff) for suff in obj_suffixes])
                obj_suffix_idx -= 1  # HACK get rid of the image size in the model name
                object_short_name = "-".join(obj_components[:obj_suffix_idx])
                model_names[idx] = object_short_name
            scene_insts = [inst for inst in scene_insts if
                           PRIM_TO_OBJECT_SHORT[inst['prim_path'].split("/")[-1]] in model_names]

        # Load object metadata
        object_metadata = self.get_model_gt_meta(self.obj_pipeline)
        # not used currently, but optional for future code extensions
        object_sdf_name = str(self.object_sdf_path)

        lib_inst = [inst for inst in
                    object_metadata['instances'].values()][0]

        return scene_cam, obj_cam, scene_insts, lib_inst

    def extract_surface_points(self,
                               cam: Cameras,
                               pipeline: Pipeline,
                               limit_box: List = None) -> TensorType[..., 3]:
        """
        Extract surface points from a given camera angle for a given model.

        Args:
            cam (Cameras): Camera view to extract points from
            pipeline (Pipeline): nerfstudio pipieline for the underlying sdf model
            limit_box (List, optional): Optional limits box to ensure no points 
            outside that box are included in output. 
            Format: [[minx,miny,minz], [maxx,maxy,maxz]]. 
            Note that ground plane removal is also included in this step 
            (using config botpts_ratio param).
            Defaults to None.

        Returns:
            TensorType[..., 3]: Tensor of 3D surface points
        """

        indices = torch.tensor([[i, j]
                                for i in range(0, cam.image_width, self.config_dict['init_sample_res'])
                                for j in range(0, cam.image_height, self.config_dict['init_sample_res'])]).long()

        raybundle = cam.generate_rays(
            0, coords=indices).to(device=pipeline.device)
        rays_packed = pipeline.model.collider(raybundle)

        # generate 3d points by running along the ray
        ray_samples, weights_list, ray_samples_list = pipeline.model.proposal_sampler(
            rays_packed, density_fns=pipeline.model.density_fns)

        # TODO should torch.no_grad() be added here?
        # Get sdf values for all points
        field_outputs = pipeline.model.field(
            ray_samples, return_alphas=True, return_occupancy=True)
        startcoords = ray_samples.frustums.get_start_positions()
        sdf_values = field_outputs[FieldHeadNames.SDF]

        # now, it is time to run the find3dpoints code
        points_3d, surfaces = uu.find3dpoints(
            sdf_values.squeeze(2), startcoords)

        # Optionally remove any points that are outside the given bounding box (far away walls etc)
        # Largely used for scene points.
        if limit_box is not None:
            # ground plane removal:
            botpts = points_3d[:, 2].min(
            ) + self.config_dict['botpts_ratio']*(points_3d[:, 2].max() - points_3d[:, 2].min())

            # remove all 3d points outside the 3d bbox of the object:
            goodidxs = []
            for idx, pt in enumerate(points_3d):
                if (limit_box[0][0] < pt[0] < limit_box[1][0]) and \
                        (limit_box[0][1] < pt[1] < limit_box[1][1]) and \
                        (limit_box[0][2] < pt[2] < limit_box[1][2]) and \
                        pt[2] > botpts:
                    goodidxs.append(idx)

            # scene_threedpts.append(this_scene_threedpts)
            return points_3d[goodidxs]
        return points_3d

    def open3dvis(self,
                  X: TensorType,
                  W: TensorType,
                  lib_threedpts: TensorType,
                  iteration: int) -> None:
        X_pcd = uu.numpy_to_pcd(
            X.clone().detach().cpu().numpy())
        W_pcd = uu.numpy_to_pcd(
            W.clone().detach().cpu().numpy())
        gt_pcd = uu.numpy_to_pcd(
            lib_threedpts.clone().detach().cpu().numpy())

        W_pcd.paint_uniform_color([0, 0, 1])  # blue
        X_pcd.paint_uniform_color([1, 0.5, 0])  # orange
        gt_pcd.paint_uniform_color([0, 1, 0])  # green

        self.openvis.add_geometry(W_pcd)
        self.openvis.add_geometry(gt_pcd)

        ctr = self.openvis.get_view_control()
        ctr.set_front([0.7, 1, 0.6])
        ctr.set_up([0, 0, 1])
        self.openvis.update_geometry(W_pcd)
        self.openvis.update_geometry(gt_pcd)
        self.openvis.poll_events()
        self.openvis.update_renderer()
        time.sleep(0.001)

        suffixname = str(self.transform_savepath).split('.')[
            0].split('/')[-1]
        suffixname = suffixname + '_' + \
            self.scene_obj_name + '_' + str(iteration) + '.png'
        fname = os.path.join(self.open3d_savepth, suffixname)
        self.openvis.capture_screen_image(fname)
        # self.openvis.remove_geometry(X_pcd)
        self.openvis.remove_geometry(W_pcd)
        self.openvis.remove_geometry(gt_pcd)

    def rotparamsetup(self, initval):
        param = nn.Parameter(torch.tensor(0.0001).to(
            self.scene_pipeline.device), requires_grad=True, ).float()
        param.data.fill_(initval)
        return param

    def main(self) -> None:
        """
        Main function for running RegNF
        """

        vis, self.openvis = self.setup_visualisations()
        scene_cam, lib_cam, insts, lib_obj = self.setup_data()
        lib_obj_name = lib_obj['prim_path'].replace('/', '')

        # Extract sample points for the library object
        # These are extracted from multiple viewpoints defined by numviews
        print(f"Extracting points from Library Object: {lib_obj_name}")

        # Calculate camera poses around the object within the scene
        # NOTE camera poses output here are in opengl format to match
        # requirements of nerfstudio camera.
        lib_cam_poses = get_obj_cam_poses(box=(lib_obj['box_min'],
                                               lib_obj['box_max']),
                                          # 0.9
                                          cam_dist=self.config_dict['camdist_library'],
                                          # 5
                                          num_poses=self.config_dict['numviews'],
                                          cam_type="opengl")['opengl']

        lib_threedpts = []
        # Go through every camera pose and extract points from the surface of the sdf
        for lib_cam_pose in lib_cam_poses:
            lib_cam.camera_to_worlds = torch.tensor(lib_cam_pose)[
                :-1, :].float()

            lcam_pts = self.extract_surface_points(lib_cam,
                                                   self.obj_pipeline)
            lib_threedpts.append(lcam_pts)

        print("Points extracted from library object")

        # Perform RegNF on all selected objects in the scene
        # NOTE this is typically just those of the same instance as the library object
        for scene_obj in insts:
            self.scene_obj_name = scene_obj['prim_path'].replace('/', '')

            print('%%%%\n Looking at Scene Object: ',
                  self.scene_obj_name, ' \n%%%%\n')

            # Calculate camera poses around the object within the scene
            # NOTE camera poses output here are in opengl format to match
            # requirements of nerfstudio camera.
            scene_cam_poses = get_obj_cam_poses(box=(scene_obj['box_min'],
                                                     scene_obj['box_max']),
                                                # 0.9
                                                cam_dist=self.config_dict['camdist_scene'],
                                                # 5
                                                num_poses=self.config_dict['numviews'],
                                                cam_type="opengl")['opengl']

            # Go through every camera pose and extract points from the surface of the sdf
            scene_threedpts = []
            for scene_cam_pose in scene_cam_poses:
                scene_cam.camera_to_worlds = torch.tensor(scene_cam_pose)[
                    :-1, :].float()

                scam_pts = self.extract_surface_points(scene_cam, self.scene_pipeline,
                                                       [scene_obj['box_min'], scene_obj['box_max']])
                scene_threedpts.append(scam_pts)

            # Convert all points to single Tensor
            scene_threedpts = torch.cat(scene_threedpts)
            lib_threedpts = torch.cat(lib_threedpts)

            print("Points extracted from scene object")

            print('Running RegNF...')
            print(f"Scene: {self.scene_obj_name}\n Library: {lib_obj_name}")

            if self.use_visdom:
                uu.visualize_points(
                    torch.cat((torch.tensor(scene_threedpts),
                              torch.tensor(lib_threedpts)), 0), vis,
                    'keys', numpts_a=scene_threedpts.shape[0], numpts_b=lib_threedpts.shape[0],
                    snum=2, title='Init guesses')

            print("Initialising ...")
            # Move objects to common [0,0,0] centroid for pose initialisation
            scene_threedpts_np = scene_threedpts.detach().cpu().numpy()  # room
            lib_threedpts_np = lib_threedpts.detach().cpu().numpy()  # object

            centroid1 = np.average(scene_threedpts_np, axis=0)
            centroid2 = np.average(lib_threedpts_np, axis=0)

            scene_threedpts_np = scene_threedpts_np - centroid1
            lib_threedpts_np = lib_threedpts_np - centroid2

            # Suitable voxel sizes from 0.001 to 0.025, may need changing based on the scale of the scene
            ransac_out, icp_out, fgr_out, pcd_scene, pcd_obj = uu.run_initial_pose_guess(
                scene_threedpts_np, lib_threedpts_np, voxel_size=self.config_dict['voxelsize'], return_fgr=True)

            tmat_icp = icp_out.transformation
            tmat_fgr = fgr_out.transformation
            pcd_scene.transform(tmat_icp)
            pose_transform = None
            # Convert back to original location and save initial transform if desired
            # TODO this is hacky as heck. Make things neater
            for tmat, tmat_name, in zip([tmat_fgr, tmat_icp], ["transforms_fgr", "transforms_init"]):
                if "fgr" in tmat_name and not self.save_fgr:
                    continue

                # Transform tmat from "zero centred" to desired form
                pose_transform = self.get_init_pose(tmat, centroid1, centroid2)

                if "init" in tmat_name and not self.save_init:
                    continue

                # If desired, save init or fgr matrix output
                # in format mirroring main transforms
                # NOTE assumes 'transforms' is part of save path
                initsavepath = str(self.transform_savepath)
                initsavepath = initsavepath.replace(
                    'transforms', tmat_name)
                if not os.path.exists(str(Path(initsavepath).parent)):
                    os.makedirs(str(Path(initsavepath).parent))
                savefilename = initsavepath.split(
                    '.')[0] + '_' + self.scene_obj_name
                np.save(savefilename, pose_transform)

            # Convert standard 4x4 matrix to parameters for nerf2nerf-style optimization
            best_translation = pose_transform[:, 3].astype(np.float32)
            R = Rotation.from_matrix(pose_transform[:3, :3])
            best_t, best_a, best_g = uu.inv_make_rot_matrix(R)

            # for a given set of 3d points, this is how we calc the surface values (signed distances):
            # get the initial surface values for RegNF optimization
            sdf_output = self.scene_pipeline.model.field.forward_geonetwork(
                scene_threedpts)
            fixed_surface_values, geo_feature = torch.split(
                sdf_output, [1, 256], dim=-1)

            fixed_surface_values = fixed_surface_values.detach().clone()

            # get the initial surface values for RegNF bi-directional optimization
            sdf_output_hat = self.obj_pipeline.model.field.forward_geonetwork(
                lib_threedpts)
            fixed_surface_values_hat, geo_feature_hat = torch.split(
                sdf_output_hat, [1, 256], dim=-1)
            fixed_surface_values_hat = fixed_surface_values_hat.detach().clone()

            # initialize network parameters and optimizer:
            thetac = self.rotparamsetup(best_t)
            if self.disable_pitch:
                alphac = torch.tensor(best_a).float().to(
                    self.scene_pipeline.device)
            else:
                alphac = self.rotparamsetup(best_a)
            if self.disable_roll:
                gammac = torch.tensor(best_g).float().to(
                    self.scene_pipeline.device)
            else:
                gammac = self.rotparamsetup(best_g)

            t = nn.Parameter(
                torch.tensor([[best_translation[0], best_translation[1], best_translation[2]]]).to(
                    self.scene_pipeline.device),
                requires_grad=True)

            scale = nn.Parameter(torch.tensor(1.000).to(self.scene_pipeline.device),
                                 requires_grad=True, ).float()

            adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                num_dims=1, float_dtype=torch.float, device=torch.device('cuda:0'), alpha_lo=1e-6, alpha_hi=1.0,
                scale_init=1)

            rotparams = [thetac]
            if not self.disable_pitch:
                rotparams.append(alphac)
            if not self.disable_roll:
                rotparams.append(gammac)

            optimizer = torch.optim.Adam(
                [{'params': rotparams, 'lr': self.config_dict['lr_rotparams']},  # 2e-2
                 {'params': [t], 'lr': self.config_dict['lr_t']},  # 1e-2
                 {'params': list(adaptive.parameters()),
                  'lr': self.config_dict['lr_ada']},  # 1e-2
                 {'params': [scale], 'lr': self.config_dict['lr_s']}])  # 1e-2

            obj_bigness = 1.0  # both are similar, refer to relative size of cad model vs query scene

            r = float(torch.norm(
                self.scene_pipeline.datamanager.train_dataset.cameras.camera_to_worlds[0][:3, 3] * scale.detach().cpu()))
            rho = r * (3 ** (0.5) / self.config_dict['rho_res'])

            X = scene_threedpts.clone().detach().to(self.scene_pipeline.device).float()
            X_hat = lib_threedpts.clone().detach().to(
                self.obj_pipeline.device).float()

            print('Running RegNF finetuning...')

            for iteration in range(self.maxiters + 1):
                # Create 4x4 transformation matrix from learnt parameters
                R = uu.make_rot_matrix(thetac, alphac, gammac)
                scale_mat = torch.eye(4).float().to(device=scale.device)
                # Don't change the next line of code. Copy operation needed for gradient computation
                scale_mat[0,0], scale_mat[1,1], scale_mat[2,2] = scale, scale, scale # grad_fn = copyslices
                rot_mat = torch.eye(4).float().to(device=scale.device)
                rot_mat[:3, :3] = R
                translate_mat = torch.eye(4).float().to(device=scale.device)
                translate_mat[:3, 3] = t
                full_transform = translate_mat @ rot_mat @ scale_mat

                # Save the current transformation matrix if desired
                if self.save_iters:
                    peritersavepath = str(self.transform_savepath)
                    peritersavepath = peritersavepath.replace(
                        'transforms', 'transforms_periter'
                    )
                    if not os.path.exists(os.path.dirname(peritersavepath)):
                        os.makedirs(os.path.dirname(
                            peritersavepath), exist_ok=True)
                    savefilename = peritersavepath.split(
                        '.')[0] + '_' + self.scene_obj_name + '_' + str(iteration) + '.npy'
                    np.save(savefilename, full_transform.detach().cpu().numpy())

                #
                W_base = torch.cat(
                    (X, torch.ones(X.shape[0], 1).to(device=scale.device)), dim=1)

                W = full_transform @ W_base.T
                W = W.T[:, :3]

                W_hat_base = torch.cat(
                    (X_hat, torch.ones(X_hat.shape[0], 1).to(device=scale.device)), dim=1)
                W_hat = torch.inverse(full_transform) @ W_hat_base.T
                W_hat = W_hat.T[:, :3]

                # Perform visualisations if desired
                if iteration % 1 == 0:
                    if self.use_visdom:
                        uu.visualize_points(torch.cat((X.clone().detach(), W.clone().detach(),
                                                       lib_threedpts), 0), vis,
                                            'keys', numpts_a=X.shape[0], numpts_b=W.shape[0],
                                            numpts_c=lib_threedpts.shape[0],
                                            snum=3, title='Sample Points')

                        uu.visualize_points(torch.cat((X_hat.clone().detach(), W_hat.clone().detach(),
                                                       scene_threedpts), 0), vis,
                                            'keys', numpts_a=X_hat.shape[0], numpts_b=W_hat.shape[0],
                                            numpts_c=scene_threedpts.shape[0],
                                            snum=3, title='Sample Points')

                    if self.use_open3d:
                        self.open3dvis(X, W, lib_threedpts, iteration)

                W = W.contiguous()  # don't remove this line. Needed for some reason
                W_hat = W_hat.contiguous()

                inputs = self.obj_pipeline.model.field.spatial_distortion(
                    W)
                sdf_output = self.obj_pipeline.model.field.forward_geonetwork(
                    inputs)  # required shape: [2043, 3]
                moving_surface_values, _ = torch.split(
                    sdf_output, [1, self.obj_pipeline.model.field.config.geo_feat_dim], dim=-1)

                inputs_hat = self.scene_pipeline.model.field.spatial_distortion(
                    W_hat)
                sdf_output_hat = self.scene_pipeline.model.field.forward_geonetwork(
                    inputs_hat)  # required shape: [2043, 3]
                moving_surface_values_hat, _ = torch.split(
                    sdf_output_hat, [1, self.scene_pipeline.model.field.config.geo_feat_dim], dim=-1)

                delta = torch.abs(moving_surface_values - fixed_surface_values)
                delta_hat = torch.abs(
                    moving_surface_values_hat - fixed_surface_values_hat)

                # W = moving surface values 3D positions (on object)
                # X_hat = fixed surface values 3D positions (on object)
                densedists = torch.cdist(W, X_hat, p=2.0)
                # regularization term over unmatched points
                totaldist = densedists.min(axis=0)[0].sum()/X_hat.shape[0]

                if (delta.mean() + delta_hat.mean()) < self.config_dict['early_stop_thresh']:
                    print('early termination')
                    break

                loss = torch.mean(adaptive.lossfun(delta))
                loss_hat = torch.mean(adaptive.lossfun(delta_hat))

                loss += loss_hat + totaldist

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration % self.config_dict['sampling_freq'] == 0 and iteration > 0:
                    newX = torch.tensor([]).to(X.device)
                    for i in range(X.shape[0]):
                        if delta[i] <= 0.2:
                            newX = torch.cat((newX, X[i].unsqueeze(0)))

                    if X_hat.shape[0] < self.config_dict['maxsamples']:
                        X_hat = self.draw_new_samples_inv(
                            X_hat, rho, torch.inverse(full_transform), obj_bigness, scale=scale)

                    X = self.draw_new_samples(
                        newX, rho, full_transform, obj_bigness, scale=scale)

                    if self.use_visdom:
                        uu.visualize_points(torch.cat((newX.clone().detach(), W.clone().detach(),
                                                       lib_threedpts), 0), vis,
                                            'keys', numpts_a=newX.shape[0], numpts_b=W.shape[0],
                                            numpts_c=lib_threedpts.shape[0],
                                            snum=3, title='Sample Points')

                    inputs = self.scene_pipeline.model.field.spatial_distortion(
                        X)
                    h = self.scene_pipeline.model.field.forward_geonetwork(
                        inputs)
                    fixed_surface_values, _ = torch.split(
                        h, [1, self.scene_pipeline.model.field.config.geo_feat_dim], dim=-1)
                    fixed_surface_values = fixed_surface_values.detach().clone()

                    inputs = self.obj_pipeline.model.field.spatial_distortion(
                        X_hat)
                    h = self.obj_pipeline.model.field.forward_geonetwork(
                        inputs)
                    fixed_surface_values_hat, _ = torch.split(
                        h, [1, self.obj_pipeline.model.field.config.geo_feat_dim], dim=-1)
                    fixed_surface_values_hat = fixed_surface_values_hat.detach().clone()

            savepath = str(self.transform_savepath)
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
            savefilename = savepath.split(
                '.')[0] + '_' + self.scene_obj_name + '.npy'
            np.save(savefilename, full_transform.detach().cpu().numpy())

            print('done an object')
            torch.cuda.empty_cache()
        if self.use_open3d:
            self.openvis.destroy_window()
        print('COMPLETE')

    def draw_new_samples(self, samples, rho, full_transform, obj_bigness, scale=1.0):
        samples = samples.unsqueeze(0)
        new_samples = uu.jitter_points(samples, rho)
        dists = uu.find_min_dist(new_samples, samples)
        smpls = torch.tensor([]).to(samples.device)

        # inputs shape = [N, 3]
        inputs = self.scene_pipeline.model.field.spatial_distortion(samples)
        h = self.scene_pipeline.model.field.forward_geonetwork(
            inputs.squeeze(0))
        surface_1_prev, _ = torch.split(
            h, [1, self.scene_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_1_prev = surface_1_prev.clone().detach().cpu().numpy()

        # in nerf2nerf, surfaces are a '1'
        # in regNF, surfaces are '0' (non-surface is >0 or <0)

        # ditch samples that are not on a valid surface
        for i in range(samples.shape[1]):  # filter previous samples
            if np.abs(surface_1_prev[i]) <= self.config_dict['surface_check_thresh']:
                smpls = torch.cat((smpls, samples[0, i].unsqueeze(0)))

        # prevent sample size from getting too big, slows down registration
        if smpls.shape[0] >= self.config_dict['maxsamples']:
            return smpls

        new_samples2 = new_samples.squeeze(0)
        W_base = torch.cat((new_samples2, torch.ones(
            new_samples2.shape[0], 1).to(device=scale.device)), dim=1)
        W = full_transform @ W_base.T
        W = W.T[:, :3]
        W = W.contiguous()

        inputs = self.scene_pipeline.model.field.spatial_distortion(
            new_samples2)
        h = self.scene_pipeline.model.field.forward_geonetwork(inputs)
        surface_1, _ = torch.split(
            h, [1, self.scene_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_1 = surface_1.clone().detach().cpu().numpy()

        inputs = self.obj_pipeline.model.field.spatial_distortion(
            W)
        h = self.obj_pipeline.model.field.forward_geonetwork(
            inputs)
        surface_2, _ = torch.split(
            h, [1, self.obj_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_2 = surface_2.clone().detach().cpu().numpy()

        delta = np.abs(surface_2 - surface_1)

        for i in range(new_samples.shape[1]):
            if dists[0, i] >= (rho * obj_bigness / 10) ** 2 and delta[i] <= self.config_dict['delta_thresh'] and np.abs(surface_1[i]) <= self.config_dict['surface_check_thresh'] and \
                    np.abs(surface_2)[i] <= self.config_dict['surface_check_thresh']:
                smpls = torch.cat((smpls, new_samples[0, i].unsqueeze(0)))

        return smpls

    def draw_new_samples_inv(self, samples, rho, full_transform, obj_bigness, scale=1.0):
        samples = samples.unsqueeze(0)
        new_samples = uu.jitter_points(samples, rho)
        dists = uu.find_min_dist(new_samples, samples)
        smpls = torch.tensor([]).to(samples.device)

        # inputs shape = [N, 3]
        inputs = self.obj_pipeline.model.field.spatial_distortion(
            samples)
        h = self.obj_pipeline.model.field.forward_geonetwork(
            inputs.squeeze(0))
        surface_1_prev, _ = torch.split(
            h, [1, self.obj_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_1_prev = surface_1_prev.clone().detach().cpu().numpy()

        # ditch samples that are not on a valid surface
        for i in range(samples.shape[1]):  # filter previous samples
            if np.abs(surface_1_prev[i]) <= self.config_dict['surface_check_thresh']:
                smpls = torch.cat((smpls, samples[0, i].unsqueeze(0)))

        # prevent sample size from getting too big, slows down registration
        if smpls.shape[0] >= self.config_dict['maxsamples']:
            return smpls

        new_samples2 = new_samples.squeeze(0)
        W_base = torch.cat((new_samples2, torch.ones(
            new_samples2.shape[0], 1).to(device=scale.device)), dim=1)
        W = full_transform @ W_base.T
        W = W.T[:, :3]
        W = W.contiguous()

        inputs = self.obj_pipeline.model.field.spatial_distortion(
            new_samples2)
        h = self.obj_pipeline.model.field.forward_geonetwork(
            inputs)
        surface_1, _ = torch.split(
            h, [1, self.obj_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_1 = surface_1.clone().detach().cpu().numpy()

        inputs = self.scene_pipeline.model.field.spatial_distortion(
            W)
        h = self.scene_pipeline.model.field.forward_geonetwork(
            inputs)
        surface_2, _ = torch.split(
            h, [1, self.scene_pipeline.model.field.config.geo_feat_dim], dim=-1)
        surface_2 = surface_2.clone().detach().cpu().numpy()

        delta = np.abs(surface_2 - surface_1)

        for i in range(new_samples.shape[1]):
            if dists[0, i] >= (rho * obj_bigness / 10) ** 2 and delta[i] <= self.config_dict['delta_thresh'] and np.abs(surface_1[i]) <= self.config_dict['surface_check_thresh'] and \
                    np.abs(surface_2)[i] <= self.config_dict['surface_check_thresh']:
                smpls = torch.cat((smpls, new_samples[0, i].unsqueeze(0)))

        return smpls


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    # tyro.cli(tyro.conf.FlagConversionOff[RegNF]).main()
    tyro.cli(RegNF).main()


if __name__ == "__main__":
    entrypoint()
