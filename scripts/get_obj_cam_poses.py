#!/usr/bin/env python
# Script for getting camera poses for point sampling in sdf2sdf

# Input
# Bounding Box of interest (tight or axis-aligned?)
# Number of poses? (default 6?)
# Distance from object for camera

# Output
# 4x4 pose of each camera w.r.t. normalised world

# NOTE will this cause issues in scene if camera is
# outside bounds of normalised world

# Create centre facing camera poses
# (optional?) rotate camera poses to match faces of the
# object bounding box
# Translate camera poses so that the new "centre" of their world
# is the centre of the object

# NOTE should find out if sampling process can become a "box"
# sampling strategy and thus we only care about points inside
# the bounding box

import math
import numpy as np
from typing import Tuple, List, Dict
from typing_extensions import Literal
import numpy.typing as npt


# TODO should be importing this from omniverse tools
# from omniverse_tools.sample_sphere import fibonacci_sphere
# Function for sampling sphere taken from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere


def fibonacci_sphere(samples=1000, scale=1, shuffle=False):
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    output = np.array(points) * scale

    # Shuffle order of points if desired
    if shuffle:
        np.random.shuffle(output)

    return output.tolist()


def switch_opencv_opengl(
        tmat: npt.NDArray
) -> npt.NDArray:
    """
    Function for converting between opencv and opengl coordinate frames.
    The same operation is done for both so both opencv to opengl and the
    reverse are supported here.

    Args:
        tmat (npt.NDArray): Original 4x4 transformation matrix to convert

    Returns:
        npt.NDArray: Converted 4x4 transformation matrix
    """
    # Transformation is simply flipping the y and z axes
    conversion_mat = np.diag([1, -1, -1, 1])
    return tmat @ conversion_mat


def get_obj_cam_poses(
        box: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        cam_dist: float = 0.5,
        num_poses: int = 6,
        cam_type: Literal["opencv", "opengl", "both"] = "opencv"
) -> Dict[str, List[npt.NDArray]]:
    """
    Function for getting sample camera poses around a given bounding box

    Args:
        box (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): 
        (box_min, box_max) bounding box min and max extent in world (x,y,z coordinates)
        cam_dist (float, optional): distance from box centre to place cameras. Defaults to 0.5.
        num_poses (int, optional): number of camera poses to get around box centre. 
        Note, if you have 6 or less views you will get a view along each axis in the following order:
        [y-, y+, x-, x+, z-, z+] (where + indicates looking along the +ve axis direction and - indicates -ve direction) 
        y is the first axis used as it currently corresponds to front/back of object (can be subject to update).
        Start with -ve so that z+ (looking from below) is last in sequence
        If not, fibonacci sphere is used. Defaults to 6.
        cam_type (str, optional): type of camera model to return. Options are "opencv", "opengl" or "both".
        "opencv" places z+ out from camera, +y pointing down on the image, and x+ pointing to the right of the image.
        "opengl" places z+ backwards from camera, +y pointing up on the image, and x+ pointing to the right of the image.
        "both" will give both with opencv stored first and opengl second.

    Raises:
        ValueError: Error if cam distance would place cameras inside the object box

    Returns:
        Dict[str, List[npt.NDArray]]: dictionary of lists of 4x4 transformation matrices as numpy arrays for each camera.
        Keys of dictionary dictate what type of camera model is being used (opencv or opengl)
    """

    box_min, box_max = np.array(box)

    # Check that camera poses won't be inside object box
    if np.any((box_max - box_min) > cam_dist):
        raise ValueError(
            f"Provided cam_dist ({cam_dist}) puts cameras inside object box")

    centroid = (box_min + box_max)/2

    # NOTE cam pose for most operations assumes opencv until the end

    # Provide fixed camera poses along axes if num_poses are less than or equal to 6
    if num_poses <= 6:
        unit_cam_coords = [[0, 1, 0], [0, -1, 0],  # facing -y and +y
                           [1, 0, 0], [-1, 0, 0],  # facing -x and +x
                           [0, 0, 1], [0, 0, -1]]  # facing -z and +z
        unit_cam_coords = unit_cam_coords[:num_poses]
    else:
        # Provide equal sampling over a sphere
        unit_cam_coords = fibonacci_sphere(num_poses)

    # Look at function adapted from
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html
    # NOTE this is a bit hacky at present. Not sure how well it would do for more
    # cooridinates
    zaxes = np.array([-1 * np.array(cam_point)
                     for cam_point in unit_cam_coords])
    xaxes = np.array([np.cross(np.array([0, 0, 1]), np.array(cam_point))
                      if not (np.allclose(cam_point, [0, 0, 1]) or
                              np.allclose(cam_point, [0, 0, -1]))
                      else np.cross(np.array([0, 1, 0]), np.array(cam_point))
                      for cam_point in unit_cam_coords])
    # else np.cross(np.array([0, 1, 0]), np.array(cam_point))
    # not (np.allclose(cam_point, [0,0,1]) or np.allclose(cam_point, [0,0,-1]))
    yaxes = np.array([np.cross(z, x) for z, x in zip(zaxes, xaxes)])

    # Create transformation matrices centred around origin
    tmats = np.array([np.eye(4) for i in range(len(unit_cam_coords))])
    # Apply camera translations
    tmats[:, :3, 3] = np.array(unit_cam_coords)*cam_dist
    # Apply camera rotations
    tmats[:, :3, :3] = np.concatenate([xaxes.reshape((-1, 3, 1)),
                                       yaxes.reshape((-1, 3, 1)),
                                       zaxes.reshape((-1, 3, 1))],
                                      axis=2)

    # Translate all camera poses to be centred around box centroid
    centroid_transform = np.eye(4)
    centroid_transform[:3, 3] = centroid
    tmats = [centroid_transform @ tmat for tmat in tmats]

    # Check if any of the camera poses now exist outside of world bounds (-1 -> 1)
    needs_rescaling = [np.any(np.abs(tmat[:3, 3]) > 1.) for tmat in tmats]
    if np.any(needs_rescaling):
        new_tmats = []
        print(f"{np.sum(needs_rescaling)} cameras need rescaling to" +
              " fit within normalised world (-1, +1)")
        for idx, tmat in enumerate(tmats):
            if not needs_rescaling[idx]:
                new_tmats.append(tmat)
                continue

            # Rescale points to exist along the same axis but a different
            # distance from the centroid of the object
            diff_vec = tmat[:3, 3] - centroid
            biggest_dim = np.argmax(np.abs(tmat[:3, 3]))
            scaling_factor = (np.abs(tmat[:3, 3][biggest_dim]) - 1) / \
                np.abs(diff_vec[biggest_dim])
            diff_vec_new = diff_vec - diff_vec*scaling_factor

            # Check if any of these end up inside the 3D bounding boxes of the objects
            # Ignore these poses if they cannot exist outside the 3D bounding box and
            # inside the world bounds
            if np.any((box_max - box_min) > np.linalg.norm(diff_vec_new)):
                continue

            # Update the translation matrix
            new_translation = centroid + diff_vec_new
            tmat[:3, 3] = new_translation
            new_tmats.append(tmat)
        if len(tmats) != len(new_tmats):
            print(
                f"After rescaling {len(tmats) - len(new_tmats)} poses could not be resolved")
            print(f"Final number of camera poses: {len(new_tmats)}")
        tmats = new_tmats

    # Transform to different camera formats if required
    if cam_type != "opencv":
        opengl_tmats = [switch_opencv_opengl(tmat) for tmat in tmats]
        if cam_type == "both":
            return {"opencv": tmats, "opengl": opengl_tmats}
        return {"opengl": opengl_tmats}

    return {"opencv": tmats}


if __name__ == "__main__":
    TEST_COORDS = ([
        -0.8823600970249572,
        -0.914437346456296,
        -0.5964213212927384
    ],
        [
        -0.26982640310835393,
        -0.3003285694171183,
        -0.12692867504095062
    ])
    cam_poses = get_obj_cam_poses(TEST_COORDS, 0.9, 6, "both")
    print(cam_poses)
