"""
This holds all miscellaneous functions used to help rendering.
Will largely include any functions that end up copying/bypassing in-built methods
"""
# general imports
from typing import Type, List, Dict
import copy
from collections import defaultdict
from pathlib import Path
import json

# torch and numpy imports
import torch
from torch import Tensor
import numpy as np
import numpy.typing as npt

# nerfstudio imports
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.models.neus_facto import NeuSFactoModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import UniformSampler

COLOUR_TO_HUE = {'red': 0, 'yellow': 60, 'green': 120,
                 'cyan': 180, 'blue': 240, 'magenta': 300}

# BELOW ADAPTED FROM https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python


def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: torch.Tensor
    :return: torch.Tensor
    """

    maxv = torch.amax(rgb, dim=1)
    maxc = torch.argmax(rgb, dim=1)
    minv = torch.amin(rgb, dim=1)
    minc = torch.argmin(rgb, dim=1)

    hsv = torch.zeros(rgb.shape, device=rgb.device)
    hsv[maxc == minc, 0] = torch.zeros(
        hsv[maxc == minc, 0].shape, device=hsv.device)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) *
                         60.0 / (maxv - minv + 1e-16)) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) *
                         60.0 / (maxv - minv + 1e-16)) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) *
                         60.0 / (maxv - minv + 1e-16)) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = torch.zeros(hsv[maxv == 0, 1].shape, device=hsv.device)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + 1e-16))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def hsv2rgb(hsv):
    """ convert HSV to RGB color space

    :param hsv: np.ndarray
    :return: np.ndarray
    """

    hi = torch.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.type(torch.uint8)
    v = hsv[..., 2].reshape(-1, 1)
    f = ((hsv[..., 0] / 60.0) - torch.floor(hsv[..., 0] / 60.0)).reshape(-1, 1)
    p = v * (1.0 - hsv[..., 1]).reshape(-1, 1)
    q = v * (1.0 - (f * hsv[..., 1].reshape(-1, 1))).reshape(-1, 1)
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1].reshape(-1, 1))).reshape(-1, 1)

    rgb = torch.zeros(hsv.shape, device=hsv.device)
    rgb[hi == 0, :] = torch.hstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = torch.hstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = torch.hstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = torch.hstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = torch.hstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = torch.hstack((v, p, q))[hi == 5, :]

    return rgb


def points_in_box(
    points: Tensor,
    box: Tensor
) -> Tensor:
    """
    Determine whether a set of 3D points exist within a given 3D box

    Args:
        points (Tensor): A tensor of query 3D points (mx3) 
        box (Tensor): A tensor of box corner coordinates (8x3).
        Note box does not have to be axis aligned

    Returns:
        Tensor: A boolean tensor of length m. 
        True represents query point is inside the box and False represents point is outside the box.
    """

    # TODO rework this, there will be a more optimal solution
    box_mins = torch.min(box, dim=0).values
    box_maxs = torch.max(box, dim=0).values
    xs_inside = torch.logical_and(points[..., 0] > box_mins[0],
                                  points[..., 0] < box_maxs[0])
    ys_inside = torch.logical_and(points[..., 1] > box_mins[1],
                                  points[..., 1] < box_maxs[1])
    zs_inside = torch.logical_and(points[..., 2] > box_mins[2],
                                  points[..., 2] < box_maxs[2])
    in_box = torch.logical_and(
        torch.logical_and(xs_inside, ys_inside), zs_inside)
    return in_box


def transform_points(
        tmat: Tensor,
        points: Tensor) -> Tensor:
    """
    Perform a 3D transformation of a set of 3D points using a
    4x4 transformation matrix. Transforming points into tmat coordinate frame.

    Args:
        tmat (Tensor): 4x4 transformation matrix tensor
        points (Tensor): mx3 tensor set of points to be transformed

    Returns:
        Tensor: mx3 tensor of transformed points
    """
    # Change mx3 into homogeneous mx4
    h_points = torch.cat(
        (points, torch.ones(points.shape[0], 1, device=points.device)), dim=1)
    # Perform transformation
    th_points = tmat @ h_points.T
    # Transform homogeneous back to mx3
    t_points = th_points.T[:, :-1]

    return t_points


def transform_ray(
        in_origins: Tensor,
        in_dvec: Tensor,
        in2out_tmat: Tensor
) -> (Tensor, Tensor):
    """


    Args:
        in_origins (_type_): _description_
        Tensor (_type_): _description_

    Returns:
        _type_: _description_
    """
    # TODO there must be a smarter way of figuring out transform directions than
    # finding origin and direction point in object world and then subtracting
    in_dpt = in_origins + in_dvec
    out_origins = transform_points(in2out_tmat,
                                   in_origins)
    out_dpt = transform_points(in2out_tmat,
                               in_dpt)
    out_dvec = out_dpt - out_origins  # not normalised
    # normalise object direction vector to account for scale
    # NOTE obj_scales should be the same for all points
    out_scales = torch.linalg.norm(
        out_dvec, dim=1)[..., None]
    out_dvec = out_dvec / out_scales
    return out_origins, out_dvec, out_scales


def get_object_outputs(
        # flat RaySamples of those for the object (no structure)
        samples: RaySamples,
        scene_to_object_tmat: Tensor,
        object_model: Type[NeuSFactoModel]

) -> Dict:
    """
    Get the outputs for an object model given samples originally in the scene's coordinate frame.

    Args:
        samples (RaySamples): Set of RaySamples in the scene's coordinate frame
        scene_to_object_tmat (Tensor): 4x4 tensor transformation matrix to go from scene to object coordinate frame
        object_model (Type[NeuSFactoModel]): Object model to provide sample outputs

    Returns:
        Dict: Outputs for the given samples from the given object model (including alphas)
        TODO provide more details.
    """
    # TODO how to apply transformation matrix simply to frustrums
    # TODO look into _apply_fin_to_dict or _apply_fn_to_fields in case they help

    # Apply transformation matrix to frustum origins and directions
    scene_origins = samples.frustums.origins  # Origin of ray
    scene_dvec = samples.frustums.directions  # Direction unit vector
    # TODO there must be a smarter way of figuring out transform directions than
    # finding origin and direction point in object world and then subtracting
    scene_dpt = scene_origins + scene_dvec  # end point of ray + direction
    object_origins = transform_points(scene_to_object_tmat,
                                      scene_origins)
    object_dpt = transform_points(scene_to_object_tmat,
                                  scene_dpt)
    object_dvec = object_dpt - object_origins  # not normalised
    # normalise object direction vector to account for scale
    # NOTE obj_scales should be the same for all points
    object_scales = torch.linalg.norm(
        object_dvec, dim=1)[..., None]
    object_dvec = object_dvec / object_scales
    # NOTE contiguous required for later functions
    samples.frustums.origins = object_origins.contiguous()
    samples.frustums.directions = object_dvec.contiguous()

    # rescale starts, ends, pixel_area by the scaling factor
    # TODO CHECK THIS IS CORRECT!!!
    samples.frustums.starts *= object_scales
    samples.frustums.ends *= object_scales
    samples.frustums.pixel_area *= object_scales * object_scales

    object_outputs = object_model.field(
        samples, return_alphas=True)

    return object_outputs


def get_object_samples(
        ray_bundle: RayBundle,
        scene_to_object_tmat: Tensor,
        object_model: Type[NeuSFactoModel]
):
    """

    Args:
        ray_bundle (RayBundle): _description_
        scene_to_object_tmat (Tensor): _description_
        object_model (Type[NeuSFactoModel]): _description_
    """
    # Going to get object samples and outputs in object frame and return in scene frame
    object_bundle = copy.deepcopy(ray_bundle)
    object_origins, object_directions, object_scales = transform_ray(ray_bundle.origins,
                                                                     ray_bundle.directions,
                                                                     scene_to_object_tmat)
    object_bundle.origins = object_origins
    object_bundle.directions = object_directions
    object_bundle.pixel_area *= object_scales*object_scales
    # TODO do I need to change the following at all
    # directions_norm

    object_samples, weights_list, ray_samples_list = object_model.proposal_sampler(
        object_bundle, density_fns=object_model.density_fns)

    object_outputs = object_model.field(
        object_samples, return_alphas=True)

    # Convert samples to scene coordinates
    num_samples_per_ray = object_samples.frustums.origins.shape[1]
    scene_origins, scene_directions, scene_scales = transform_ray(object_origins,
                                                                  object_directions,
                                                                  torch.linalg.inv(scene_to_object_tmat))

    scene_origins = scene_origins.unsqueeze(
        1).repeat(1, num_samples_per_ray, 1)
    scene_directions = scene_directions.unsqueeze(
        1).repeat(1, num_samples_per_ray, 1)
    # scene_scales = scene_scales.unsqueeze(1).repeat(
    #     1, num_samples_per_ray, 1)
    object_samples.frustums.origins = scene_origins.contiguous()
    object_samples.frustums.directions = scene_directions.contiguous()
    object_samples.frustums.starts *= scene_scales[..., None]  # num_raysx1x1
    object_samples.frustums.ends *= scene_scales[..., None]
    # TODO investigate why in place multiplication fails only for this sample
    object_samples.frustums.pixel_area = object_samples.frustums.pixel_area * \
        scene_scales[..., None] ** 2

    return object_samples, object_outputs


def change_colour(rgbs, colour):
    replace_hsvs = rgb2hsv(rgbs)
    if isinstance(colour, str):
        replace_hue = COLOUR_TO_HUE[colour]
    elif isinstance(colour, int):
        replace_hue = colour
    else:
        raise ValueError(
            f"object colour must be string or int: {colour}")

    replace_hsvs[..., 0] = replace_hue
    replace_hsvs[..., 1] = 1
    return hsv2rgb(replace_hsvs)


def _merge_frustums_components(frustums1, frustums2, component, all_sample_order):
    return __merge_sample_data(getattr(frustums1, component),
                               getattr(frustums2, component),
                               all_sample_order)


def __merge_sample_data(tensor1, tensor2, all_sample_order):
    # TODO There should be better way than this
    # Scatter requires idx where src should go. Have idx of source for location.
    # Fix through a secondary argsort operation.
    source_data = torch.cat((tensor1, tensor2), dim=1)
    idxs = torch.argsort(all_sample_order, dim=1).repeat(
        1, 1, source_data.shape[2])
    merged_data = torch.zeros_like(source_data)
    merged_data.scatter_(1, idxs, source_data)
    return merged_data


def merge_raysamples(ray_samples, new_o_samples, all_sample_order):

    # Create new set of Frustrums
    # WARNING!!! Seems dodgy. Frustums are likely overlapping.
    # This could screw up depth image generation
    ray_frustrums = Frustums(
        origins=_merge_frustums_components(ray_samples.frustums,
                                           new_o_samples.frustums,
                                           "origins",
                                           all_sample_order),
        directions=_merge_frustums_components(ray_samples.frustums,
                                              new_o_samples.frustums,
                                              "directions",
                                              all_sample_order),
        starts=_merge_frustums_components(ray_samples.frustums,
                                          new_o_samples.frustums,
                                          "starts",
                                          all_sample_order),
        ends=_merge_frustums_components(ray_samples.frustums,
                                        new_o_samples.frustums,
                                        "ends",
                                        all_sample_order),
        pixel_area=_merge_frustums_components(ray_samples.frustums,
                                              new_o_samples.frustums,
                                              "pixel_area",
                                              all_sample_order),
    )
    # TODO implement the following set of stuff if we run into issues
    # Merge Cameras
    # Merge deltas
    # WARNING!!! This seems supremely dodgy. deltas should be correct for
    # each sample as it was taken and used, but have no relation to eachother.
    # Merge spacing info (starts, ends, euclidean function)
    return RaySamples(ray_frustrums)


def merge_outputs(outputs1, outputs2, all_sample_order):
    # TODO HACKY AS HELL. SDF for new points should change based on underlying models
    merged_outputs = {}
    for key, val in outputs1.items():
        merged_outputs[key] = __merge_sample_data(
            val,
            outputs2[key],
            all_sample_order
        )
    return merged_outputs


def sample_and_forward_fields(
        ray_bundle: RayBundle,
        scene_model: Type[NeuSFactoModel],
        object_models: List[Type[NeuSFactoModel]],
        scene_remove_boxes: List[Tensor],
        object_replace_boxes: List[Tensor],
        object_transforms: List[Tensor],
        object_colours: List,
        sampler: str = 'uniform'
) -> Dict:
    """
    Replacement for the in-built sample and forward fields of neus_facto.py
    NOTE WORK IN PROGRESS!! 
    Currently using in-built sampler

    TODO Update to better sampling strategy (perhaps uniform to start)
    Currently only going to care about the final outputs of the sampler, not the priors

    # TODO in future replace multiple object lists with object dictionary

    Args:
        ray_bundle (RayBundle): Bundle of rays being sampled
        scene_model (Type[NeuSFactoModel]): Scene model to use for most rendering
        object_models (List[Type[NeuSFactoModel]]): Object models to replace outlined sections of the scene
        scene_remove_boxes (List[Tensor]): list of 8x3 box corners for regions that are to be 
        removed from the scene model. Must be in scene model coordinates.
        object_replace_boxes (List[Tensor]): list of 8x3 box corners for regions that are to 
        have scene model replaced by object models. Must be in scene model coordinates. 
        Order must align with object models
        object_transforms (List[Tensor]): list of 4x4 transformation matrices that transform points from scene 
        to object model coordinates. Order must align with object_models list
        object_colours (List): list of object colours to be applied to object replacements. None = ignored. Order must align with object_models list.

    Returns:
        Dict: Dictionary of outputs from the sampling and querying of sdf fields. 

        Dictionary keys are: 
         - "ray_samples" -> RaySamples for all of the queried rays in the ray_bundle
         - "field_outputs"-> The full set of outputs for all of the ray_samples queried,
         - "weights" -> The full set of weights to use for all samples in rendering (transmittance * sample alpha),
         - "bg_transmittance" -> The level of transmittance before the background model is reached,
    """

    # TODO have some way of merging these two together or ... something
    # Use in-built sampler for the scene to extract samples along the ray

    if sampler == 'uniform':
        new_sampler = UniformSampler(200)
        ray_samples = new_sampler.generate_ray_samples(ray_bundle)
        add_o_samples = False
    elif sampler == "original":
        ray_samples, weights_list, ray_samples_list = scene_model.proposal_sampler(
            ray_bundle, density_fns=scene_model.density_fns)
        add_o_samples = True
    else:
        raise ValueError(
            f"Invalid sampler provided: '{sampler}'. Valid options are 'uniform' and 'original'")

    # TODO investigate what impact the frustrums will have (when between scene and object or removed boundary)
    sample_points = ray_samples.frustums.get_positions()

    # Determine which points should be removed (become transparent). Defaults to none
    scene_remove_samples = torch.zeros((sample_points.shape[0], sample_points.shape[1]),
                                       dtype=bool, device=scene_model.device)
    if scene_remove_boxes is not None:
        for remove_box in scene_remove_boxes:
            in_box = points_in_box(sample_points, remove_box)
            scene_remove_samples = torch.logical_or(
                scene_remove_samples, in_box)

    # Determine which samples should be replaced by object models and which ones (indicated by idx of object model +1).
    # Defaults to none (all zeros)
    scene_replace_samples = torch.zeros(
        (sample_points.shape[0], sample_points.shape[1]), dtype=int, device=scene_model.device)
    if object_replace_boxes is not None:
        for idx, replace_box in enumerate(object_replace_boxes):

            in_box = points_in_box(sample_points, replace_box)

            # TODO handle outlier where multiple replace boxes overlap
            # Currently takes the last index applied to it

            # add one to index to allow for unchanged points (value zero)
            scene_replace_samples[in_box] = idx+1

    # NOTE fairly confident scene distortions should do nothing here for now.
    # Scene distortions are for when sampling outside of normalised box
    field_outputs = scene_model.field(
        ray_samples, return_alphas=True)
    # set alphas to zero where removing samples.
    # TODO find out whether alphas should be effected by changing of underlying scene
    # TODO update to not change alphas but density and other associated values.
    field_outputs[FieldHeadNames.ALPHA][scene_remove_samples] = 1e-12

    # Set alphas and colours to object model values where replacing samples
    # TODO surely there is a quicker way than the for loop approach
    for object_id in torch.unique(scene_replace_samples):
        # Skip unchanged pixels
        if object_id == 0:
            continue
        # Get the samples to be fed into other networks
        object_mask = scene_replace_samples == object_id

        # TODO find out why this looks so attrocious when actually querying network at given locations
        # For original sampling strategy
        if not add_o_samples:
            # flat RaySamples of those for the object (no structure)
            object_samples = copy.deepcopy(ray_samples[object_mask])

            replace_field_outputs = get_object_outputs(samples=object_samples,
                                                       scene_to_object_tmat=object_transforms[object_id-1],
                                                       object_model=object_models[object_id-1])

            replace_rgbs = replace_field_outputs[FieldHeadNames.RGB]
            if object_colours is not None and len(object_colours) > 0 and object_colours[object_id-1] is not None:
                replace_rgbs = change_colour(
                    replace_rgbs, object_colours[object_id-1])

            # Replace the rgb and alpha values in main field outputs with replacements
            # TODO in future change more/different to just ALPHA and RGB
            field_outputs[FieldHeadNames.ALPHA][object_mask] = replace_field_outputs[FieldHeadNames.ALPHA]
            field_outputs[FieldHeadNames.RGB][object_mask] = replace_rgbs

        # Add samples from object-specific sampler
        # NOTE we will do this for all rays even if object is only present in some
        # rays because of data format for ray samples
        if add_o_samples:
            # Set alphas for old points in the region to nothing (don't want them contributing to the picture)
            field_outputs[FieldHeadNames.ALPHA][object_mask] = 1e-12
            # TODO naming convention tidy
            # Get new samples in scene coordinates and matching object outputs
            new_o_samples, new_o_outputs = \
                get_object_samples(ray_bundle,
                                   scene_to_object_tmat=object_transforms[object_id-1],
                                   object_model=object_models[object_id-1])
            # Change colour output if required
            if object_colours is not None and len(object_colours) > 0 and object_colours[object_id-1] is not None:
                old_shape = new_o_outputs[FieldHeadNames.RGB].shape
                new_o_outputs[FieldHeadNames.RGB] = change_colour(
                    new_o_outputs[FieldHeadNames.RGB].reshape(-1, 3), object_colours[object_id-1]).reshape(old_shape)

            # Find what samples should be ignored due to not being in the scene object region
            # Set their outputs to near-zero alpha
            new_out_box = torch.logical_not(points_in_box(
                new_o_samples.frustums.get_positions(), object_replace_boxes[object_id-1]))
            new_o_outputs[FieldHeadNames.ALPHA][new_out_box] = 1e-12

            # Find where new samples should lie along the ray
            all_sample_distances = torch.cat(
                (ray_samples.frustums.starts, new_o_samples.frustums.starts), dim=1)
            all_sample_order = torch.argsort(all_sample_distances, dim=1)

            # Merge all samples into new final bundle
            ray_samples = merge_raysamples(
                ray_samples, new_o_samples, all_sample_order)
            # Merge all outputs into new final values
            field_outputs = merge_outputs(
                field_outputs, new_o_outputs, all_sample_order)

    # NOTE this is all an absolute hack, this won't help recover the SDF of the
    # scene plus new objects in any meaningful way. This is just for rendering.
    weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
        field_outputs[FieldHeadNames.ALPHA]
    )
    bg_transmittance = transmittance[:, -1, :]

    # TODO double check how/if I can remove these lists
    # weights_list.append(weights)
    # ray_samples_list.append(ray_samples)

    samples_and_field_outputs = {
        "ray_samples": ray_samples,
        "field_outputs": field_outputs,
        "weights": weights,
        "bg_transmittance": bg_transmittance,
        # "weights_list": weights_list,
        # "ray_samples_list": ray_samples_list,
    }
    return samples_and_field_outputs


def render_outputs(
        ray_bundle: RayBundle,
        samples_and_field_outputs: Dict,
        scene_model: Type[NeuSFactoModel]
) -> Dict:
    """
    Function to take ray bundle and outputs for scene model and 
    get final render outputs for each ray.
    NOTE originally copied from base_surface_model.py but separated here
    to allow for prior changes to sample and field outputs

    Args:
        ray_bundle (RayBundle): bundle of rays to use for rendering
        samples_and_field_outputs (Dict): outputs from network/s defining properties of samples along rays
        scene_model (Type[NeuSFactoModel]): Scene model to use for rendering process

    Returns:
        Dict: Dictionary of the rendered outputs (including background outputs) for the ray bundle
    """
    field_outputs = samples_and_field_outputs["field_outputs"]
    ray_samples = samples_and_field_outputs["ray_samples"]
    weights = samples_and_field_outputs["weights"]
    bg_transmittance = samples_and_field_outputs["bg_transmittance"]

    # TODO double check standard render and whether it will likely be screwed
    # by anything we have done so far
    rgb = scene_model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB],
                                   weights=weights)
    depth = scene_model.renderer_depth(weights=weights,
                                       ray_samples=ray_samples)
    # the rendered depth is point-to-point distance and we should convert to depth
    # NOTE this means depth to image plane I believe
    depth = depth / ray_bundle.directions_norm

    # TODO figure out if our changing of weights should also change NORMAL values at all ...
    normal = scene_model.renderer_normal(
        semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
    accumulation = scene_model.renderer_accumulation(
        weights=weights)

    # background model code unchanged
    if scene_model.config.background_model != "none":
        # TODO remove hard-coded far value
        # sample inversely from far to 1000 and points and forward the bg model
        ray_bundle.nears = ray_bundle.fars
        ray_bundle.fars = torch.ones_like(
            ray_bundle.fars) * scene_model.config.far_plane_bg

        ray_samples_bg = scene_model.sampler_bg(ray_bundle)
        # use the same background model for both density field and occupancy field
        field_outputs_bg = scene_model.field_background(
            ray_samples_bg)
        weights_bg = ray_samples_bg.get_weights(
            field_outputs_bg[FieldHeadNames.DENSITY])

        rgb_bg = scene_model.renderer_rgb(
            rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
        depth_bg = scene_model.renderer_depth(
            weights=weights_bg, ray_samples=ray_samples_bg)
        accumulation_bg = scene_model.renderer_accumulation(
            weights=weights_bg)

        # merge background color to forgound color
        rgb = rgb + bg_transmittance * rgb_bg

        bg_outputs = {
            "bg_rgb": rgb_bg,
            "bg_accumulation": accumulation_bg,
            "bg_depth": depth_bg,
            "bg_weights": weights_bg,
        }
    else:
        bg_outputs = {}
    outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
        "normal": normal,
        "weights": weights,
        "ray_points": scene_model.scene_contraction(
            ray_samples.frustums.get_start_positions()
        ),  # used for creating visiblity mask
        # used to scale z_vals for free space and sdf loss
        "directions_norm": ray_bundle.directions_norm,
    }
    outputs.update(bg_outputs)

    # this is used only in viewer
    outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
    return outputs


def render_scene_img(
        scene_raybundle: RayBundle,
        scene_model: Type[NeuSFactoModel],
        scene_remove_boxes: List[Tensor] = None,
        object_models: List[NeuSFactoModel] = None,
        object_replace_boxes: List[Tensor] = None,
        object_transforms: List[Tensor] = None,
        object_colours: List = None,
        sampler: str = "original"
) -> npt.NDArray:
    """
    Render an image of a scene given a scene's raybundle and any editing operations to be applied.
    Editing operations supported are removing regions of the scene and replacing regions of a scene
    with another neural model's outputs.

    TODO replace all the different object model, replace boxes, etc. with a dictionary of values

    NOTE largely copied from base_model.py with some extra comments added

    Args:
        scene_raybundle (RayBundle): RayBundle of all rays for the scene image in scene coordinate frame
        scene_model (Type[NeuSFactoModel]): Model for the baseline scene to be rendered (core to do edits around)
        scene_remove_boxes (List[Tensor], optional): List of regions to remove from scene, 
        defined by 8x3 tensors of bounding box corners. Defaults to None.
        object_models (List[NeuSFactoModel], optional): List of object models to use for replacement edit operation. 
        Defaults to None.
        object_replace_boxes (List[Tensor], optional): List of regions to use for replacing objects within the 
        scene's coordinate frame. 
        Defined by 8x3 tensors of bounding box corners. Order must align with object models. Defaults to None.
        object_transforms (List[Tensor], optional): List of transformations to go from scene to object coordinate frame 
        provided as 4x4 transformation matrix Tensors. Order must align with object models. Defaults to None.

    Returns:
        npt.NDArray: Numpy array of the rendered image HxWx3
    """
    with torch.no_grad():
        # Divvy up the image into chunks
        num_rays_per_chunk = scene_model.config.eval_num_rays_per_chunk
        image_height, image_width = scene_raybundle.origins.shape[:2]
        num_rays = len(scene_raybundle)
        outputs_lists = defaultdict(list)
        # For each chunk of the image
        for i in range(0, num_rays, num_rays_per_chunk):
            # Grab the rays associated with the chunk
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = scene_raybundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx)

            # Get the near and far points for the ray bundle using collider
            if scene_model.collider is not None:
                ray_bundle = scene_model.collider(ray_bundle)

            # Figure out the samples and outputs for the rays (including edits)
            samples_and_field_outputs = sample_and_forward_fields(
                ray_bundle=ray_bundle,
                scene_model=scene_model,
                object_models=object_models,
                scene_remove_boxes=scene_remove_boxes,
                # NOTE should be in scene coordinates already
                object_replace_boxes=object_replace_boxes,
                object_transforms=object_transforms,
                object_colours=object_colours,
                sampler=sampler
            )
            # Render outputs along each ray
            scene_out = render_outputs(
                ray_bundle=ray_bundle,
                samples_and_field_outputs=samples_and_field_outputs,
                scene_model=scene_model
            )
            for output_name, output in scene_out.items():  # type: ignore
                outputs_lists[output_name].append(output)
        scene_out = {}
        # Concatenate all of the ray chunks into a single image
        for output_name, outputs_list in outputs_lists.items():
            # TODO see if there is a better fix than saying "only care about rgb here"
            if not torch.is_tensor(outputs_list[0]) or output_name != "rgb":
                # TODO: handle lists of tensors as well
                continue
            scene_out[output_name] = torch.cat(outputs_list).view(
                image_height, image_width, -1)  # type: ignore
    # Turn image into a numpy array (range 0-255)
    scene_img = (scene_out["rgb"].cpu().numpy()
                 * 256).clip(0, 255).astype(np.uint8)
    return scene_img


def get_model_gt_meta(pipeline):
    # TODO sort out typing for pipeline
    data_path = Path(pipeline.datamanager.config.dataparser.data)
    with open(data_path / "meta_data.json", "r") as f:
        gt_meta = json.load(f)
    return gt_meta


def new_obj_s2o_tmat(
        old_tmat: torch.Tensor,
        old_meta: Dict,
        new_meta: Dict,

) -> torch.Tensor:
    """
    Updates scene to object transformation matrix for a new object.
    Will assume that we want origins of models in global coordinates to align
    and that these origins are located at the base of objects that are oriented.

    NOTE assumption can be loosened in future if desired

    Args:
        old_tmat (torch.Tensor): The original 4x4 transformation matrix
        going from scene coordinates to the centre of the original library object's 
        coordinate frame (assumed essentially middle of object).
        old_meta (Dict): Metadata for the original object (from data used to train model)
        new_meta (Dict): Metadata for the nerw object (from data used to train model)

    Returns:
        torch.Tensor: updated 4x4 transformation matrix going from scene coordinates
        to the centre of the new library object's coordinate frame (assumed essentially middle of object)
    """
    # WARNING assumption that all objects had origins in the same place was NOT CORRECT!!
    # WIP trying to fix it so it doesn't matter, so long as we assume same orientation and that we are in the centre.

    # TODO update to make look neater

    # Can I just replace old_base_to_gto and gto_to_new_base
    old_base_to_owo = torch.eye(4, device=old_tmat.device)
    old_base_to_owo[2, 3] = -1 * old_meta["instances"]["1"]["box_min"][2]
    new_base_to_nwo = torch.eye(4, device=old_tmat.device)
    new_base_to_nwo[2, 3] = -1 * new_meta["instances"]["1"]["box_min"][2]
    old_world_to_gt = torch.tensor(
        old_meta["worldtogt"], device=old_tmat.device)
    new_world_to_gt = torch.tensor(
        new_meta["worldtogt"], device=old_tmat.device)
    ob_to_gto = old_world_to_gt @ old_base_to_owo
    nb_to_gto = new_world_to_gt @ new_base_to_nwo
    hacky_diff = torch.eye(4, device=old_tmat.device)
    hacky_diff[:3, -1] = ob_to_gto[:3, -1] - nb_to_gto[:3, -1]
    new_tmat = torch.linalg.inv(
        new_world_to_gt) @ hacky_diff @ old_world_to_gt @ old_tmat
    return new_tmat
