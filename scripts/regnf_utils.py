import torch
import random
import socket
import json
import numpy as np
import open3d as o3d
import os
import yaml
from pathlib import Path
# import visdom


def custom_sampler(image_batch, mask=None, n=4):
    device = image_batch["image"].device
    num_images, image_height, image_width, _ = image_batch["image"].shape
    # indices needs a shape of (N, 3), where N is the number of rays

    if mask is not None:
        nonzero_indices = torch.nonzero(
            mask[..., 0].to(device), as_tuple=False)
        if len(nonzero_indices) > 10000:
            chosen_indices = random.sample(
                range(len(nonzero_indices)), k=10000)
            indices = nonzero_indices[chosen_indices]
        else:
            indices = nonzero_indices
    else:
        indices = torch.tensor([[0, i, j]
                                for i in range(0, image_width, n)
                                for j in range(0, image_height, n)]).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch = {
        key: value[c, y, x]
        for key, value in image_batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (
        indices.shape[0], 3), collated_batch["image"].shape

    if "sparse_sfm_points" in image_batch:
        collated_batch["sparse_sfm_points"] = image_batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = image_batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices
    return collated_batch


def find3dpoints(sdf, pts):
    values = []
    idxs = []
    for i in range(sdf.shape[0]):
        if sdf[i, :].min() > 0:
            values.append(1.0)
            idxs.append(-1)
        elif sdf[i, :].max() < 0:
            values.append(1.0)
            idxs.append(-1)
        else:
            zerocrossings = torch.where(torch.diff(torch.sign(sdf[i, :])))[0]

            # first crossing:
            idx = zerocrossings[0]
            value = sdf[i, idx]
            values.append(value)
            idxs.append(idx)

    values = torch.tensor(values)
    idxs = torch.tensor(idxs)

    # values, idx = torch.abs(sdf).min(dim=1)

    # todo: experiment and see if this is still needed
    realsurfaces = (values < 0.01)
    # goodvalues = values[realsurfaces]
    goodidx = idxs[realsurfaces]
    threedpts = pts[realsurfaces, goodidx, :]

    return threedpts, realsurfaces


def visualize_points(points, vis, win_name,
                     numpts_a=0, numpts_b=0, numpts_c=0,
                     title="Sample Points", change_color=True, snum=0):
    num = points.shape[0]
    points = points.view(num, 3)
    b = 2 if change_color else 1
    c = 3 if change_color else 1
    Ys = torch.tensor([1] * (numpts_a) + [b] * (numpts_b) + [c] * (numpts_c))

    # Ys_test = torch.tensor([2] * (numpts_a))
    # points_test = points[:numpts_a]

    # todo: vis centroids
    vis.scatter(
        X=points,
        Y=Ys,
        win=win_name,
        opts=dict(
            title=title,
            markersize=4,
            xtickmin=-1,
            xtickmax=1,
            xtickstep=0.1,
            ytickmin=-1,
            ytickmax=1,
            ytickstep=0.1,
            ztickmin=-1,
            ztickmax=1,
            ztickstep=0.1)
    )


# def rot2eul(R):
#     beta = -np.arcsin(R[2,0])
#     alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
#     gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
#     return np.array((alpha, beta, gamma))


def inv_make_rot_matrix(R):
    # transforms from rotatio matrix to network rotation parameters
    eulers = R.as_euler('xyz')  # zyx # theta, alpha, gamma

    if eulers[2] < 0:
        thetac = torch.tensor(eulers[2]) + (2 * torch.pi)
    else:
        thetac = torch.tensor(eulers[2])
    thetac = thetac / (2 * torch.pi)
    thetac = torch.logit(thetac)

    if eulers[1] < 0:
        alphac = torch.tensor(eulers[1]) + (2 * torch.pi)
    else:
        alphac = torch.tensor(eulers[1])
    alphac = alphac / (2 * torch.pi)
    alphac = torch.logit(alphac)

    if eulers[0] < 0:
        gammac = torch.tensor(eulers[0]) + (2 * torch.pi)
    else:
        gammac = torch.tensor(eulers[0])
    gammac = gammac / (2 * torch.pi)
    gammac = torch.logit(gammac)
    return thetac, alphac, gammac


def make_rot_matrix(thetac, alphac, gammac, device=torch.device('cuda')):
    # transforms from network rotation parameters to rotation matrix
    theta = thetac.sigmoid() * torch.pi * 2
    alpha = alphac.sigmoid() * torch.pi * 2
    gamma = gammac.sigmoid() * torch.pi * 2
    m1 = torch.cat((torch.cat((theta.cos().unsqueeze(0),  # yaw (rotation about z axis)
                               -theta.sin().unsqueeze(0), torch.tensor([0.0]).to(device))).unsqueeze(0),
                    torch.cat((theta.sin().unsqueeze(0), theta.cos().unsqueeze(0),
                               torch.tensor([0.0]).to(device))).unsqueeze(0),
                    torch.tensor([[0.0, 0.0, 1.0]]).to(device)), 0).float()

    m2 = torch.cat((torch.cat((alpha.cos().unsqueeze(0), torch.tensor([0.0]).to(device),  # pitch (rotation about y axis)
                               alpha.sin().unsqueeze(0))).unsqueeze(0),
                    torch.tensor([[0.0, 1.0, 0.0]]).to(device),
                    torch.cat((-alpha.sin().unsqueeze(0), torch.tensor([0.0]).to(device),
                               alpha.cos().unsqueeze(0))).unsqueeze(0),
                    ), 0).float()

    m3 = torch.cat(
        (torch.tensor([[1.0, 0.0, 0.0]]).to(device), torch.cat((torch.tensor([0.0]).to(device),  # roll (rotation about x axis)
                                                                gamma.cos().unsqueeze(0),
                                                                -gamma.sin().unsqueeze(0))).unsqueeze(0),
         torch.cat((torch.tensor([0.0]).to(device), gamma.sin().unsqueeze(0),
                    gamma.cos().unsqueeze(0))).unsqueeze(0),
         ), 0).float()

    R = (m1 @ m2 @ m3)  # xyz (backwards)
    return R


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def jitter_points(samples, d):  # [1,n,3]
    noise = (torch.rand(samples.shape) - 0.5) * 2
    norm = torch.sqrt(torch.sum(noise ** 2, axis=-1)).unsqueeze(-1)
    noise = noise / norm * d
    return samples + noise.to(samples.device)


def find_min_dist(new_set, samples):  # [1,n,3], [1,n,3]

    n = samples.shape[1]
    nn = new_set.unsqueeze(1).repeat((1, n, 1, 1))
    ss = samples.unsqueeze(-2).repeat((1, 1, n, 1))
    return torch.min(torch.sum((nn - ss) ** 2, -1), axis=1)[0]  # [1,n]


def load_nerf2nerf_keypoints(file_p, nerf2nerf_root):
    with open(file_p) as file:
        t_dict = json.load(file)

    if t_dict["keypoints"] == 0:
        return None
    As = []
    A_exts = []
    for i in t_dict["keypoints"]['1'].keys():
        As += [np.array(t_dict["keypoints"]['1'][i])]
        with open(os.path.join(nerf2nerf_root, os.path.normpath(i))) as f:
            ext_dict = json.load(f)
        A_exts += [torch.tensor(
            np.array(ext_dict["transform_matrix"], dtype=np.float64))]

    Bs = []
    B_exts = []
    for i in t_dict["keypoints"]['2'].keys():
        Bs += [np.array(t_dict["keypoints"]['2'][i])]
        with open(os.path.join(nerf2nerf_root, os.path.normpath(i))) as f:
            ext_dict = json.load(f)
        B_exts += [torch.tensor(
            np.array(ext_dict["transform_matrix"], dtype=np.float64))]
    return As, Bs, A_exts, B_exts


def load_poses_nerf2nerf(file, res):
    raw_W, raw_H = res
    meta_fname = file
    with open(meta_fname) as file:
        meta = json.load(file)
    p_list = meta["frames"]
    focal = 0.5 * raw_W / np.tan(0.5 * meta["camera_angle_x"])
    pose_raw_all = [torch.tensor(
        f["transform_matrix"], dtype=torch.float32) for f in p_list]
    # why does nerf2nerf flip the rotations??
    pose_all = torch.stack([parse_raw_camera_nerf2nerf(p)
                           for p in pose_raw_all], dim=0)
    return focal, pose_all


def parse_raw_camera_nerf2nerf(pose_raw):  # extrinsics
    R = torch.diag(torch.tensor([1, -1, -1]))
    t = torch.zeros(R.shape[:-1], device=R.device)
    pose_flip = torch.cat(
        [torch.cat([R, t[..., None]], dim=-1), torch.tensor([[0., 0., 0., 1]])], dim=0)
    pose = (pose_raw @ pose_flip)[..., :3, :]
    return pose


def get_intrinsic_nerf2nerf(focal, image_size=(512, 512), device=torch.device('cuda')):
    intrinsic = torch.tensor([
        [focal, 0., image_size[0] / 2, 0],
        [0., focal, image_size[1] / 2, 0],
        [0., 0., 1, 0],
        [0., 0., 0, 1]]).float()
    return intrinsic


def triangulate_nerf2nerf(A1, A2, extrinsic1, extrinsic2, focal, image_size=(512, 512)):
    num = A1.shape[0]
    intrinsic = get_intrinsic_nerf2nerf(
        focal, image_size=image_size).detach().clone().cpu().numpy()
    intrinsici = np.linalg.inv(intrinsic)
    # extrinsic1 = extrinsic1[0].numpy()
    # extrinsic2 = extrinsic2[0].numpy()

    roA = np.array([extrinsic1[:, 3]]).T
    roB = np.array([extrinsic2[:, 3]]).T

    rdAs = []
    rdBs = []
    for i in range(num):
        rdA = extrinsic1 @ intrinsici @ np.array(
            [[A1[i, 0], A1[i, 1], 1, 1]]).T - roA
        rdA = rdA / np.linalg.norm(rdA)
        rdA = rdA[:3]
        rdAs += [rdA]
        rdB = extrinsic2 @ intrinsici @ np.array(
            [[A2[i, 0], A2[i, 1], 1, 1]]).T - roB
        rdB = rdB / np.linalg.norm(rdB)
        rdB = rdB[:3]
        rdBs += [rdB]

    roB = roB[:3]
    roA = roA[:3]

    As = []
    for i in range(num):
        tA = np.dot(np.cross((roB.T[0] - roA.T[0]), rdBs[i].T[0]), np.cross(rdAs[i].T[0], rdBs[i].T[0])) / np.dot(
            (np.cross(rdAs[i].T[0], rdBs[i].T[0])), (np.cross(rdAs[i].T[0], rdBs[i].T[0])))
        PA = roA + tA * rdAs[i]

        tB = np.dot(np.cross((roA.T[0] - roB.T[0]), rdAs[i].T[0]), np.cross(rdBs[i].T[0], rdAs[i].T[0])) / np.dot(
            (np.cross(rdBs[i].T[0], rdAs[i].T[0])), (np.cross(rdBs[i].T[0], rdAs[i].T[0])))
        PB = roB + tB * rdBs[i]

        As += [(PA + PB) / 2]
    return torch.tensor(np.array(As))


def parse_raw_camera_nerf2nerf(pose_raw):  # extrinsics
    R = torch.diag(torch.tensor([1, -1, -1]))
    t = torch.zeros(R.shape[:-1], device=R.device)
    pose_flip = torch.cat(
        [torch.cat([R, t[..., None]], dim=-1), torch.tensor([[0., 0., 0., 1]])], dim=0)
    pose = (pose_raw @ pose_flip)[..., :3, :]
    return pose


def invert_nerf2nerf(pose):
    # invert a camera pose
    R, t = pose[..., :3], pose[..., 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = (-R_inv @ t)[..., 0]
    pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)
    return pose_inv


def generate_camera_point_spherical_nerf2nerf(r, theta=None, phi=None, phi_low=np.pi / 4, r_scale=1., z_offset=0.):
    phi_low = np.pi / 4 if phi_low is None else phi_low
    r_scale = 1. if r_scale is None else r_scale
    z_offset = 0.2 if z_offset is None else z_offset
    r = r * r_scale
    if theta is None and phi is None:
        theta, phi = np.random.rand() * 2 * np.pi, np.random.uniform(low=phi_low,
                                                                     high=np.pi / 2)

    z = np.sin(phi) * r
    xy = r * np.cos(phi)
    y = np.cos(theta) * xy
    x = np.sin(theta) * xy

    forward = np.array([-x, -y, z_offset - z])
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rot = np.stack([right, up, -forward], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = np.array([x, y, z])
    pose = parse_raw_camera_nerf2nerf(torch.tensor(pose).float())
    pose = torch.concat(
        (pose, torch.tensor([[0, 0, 0.0, 1]]).to(pose.device)), axis=0)

    return pose


# make a random view on the sphere
def make_random_extrinsic_nerf2nerf(sample, r_scale=1, z_offset=0, phi_low=np.pi / 4):
    sample = invert_nerf2nerf(sample)
    r = torch.norm(sample[:3, 3])
    return generate_camera_point_spherical_nerf2nerf(r, r_scale=r_scale, z_offset=z_offset, phi_low=phi_low)

# initial pose guess functions - credit to Sutharsan


def numpy_to_pcd(numpy_array):
    """
    Convert a numpy array to a .pcd file and save it.

    :param numpy_array: Numpy array of shape (N, 3), where N is the number of points.
    :param save_path: Path to save the .pcd file.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy_array)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("data/objectpts_fancychair.pcd")
    # target = o3d.io.read_point_cloud("data/scenepts_fancychair.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            4000000, 500),
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ])

    return result


def refine_registration(result_ransac, source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(source=source,
                                                         target=target,
                                                         max_correspondence_distance=distance_threshold,
                                                         init=result_ransac.transformation,
                                                         )

    return result


def run_initial_pose_guess(scene_threedpts_np, obj_threedpts_np, voxel_size=0.05, return_fgr=False):
    scene_threedpts = numpy_to_pcd(scene_threedpts_np)
    obj_threedpts = numpy_to_pcd(obj_threedpts_np)

    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(scene_threedpts, obj_threedpts, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)

    result_icp = refine_registration(result_ransac, source, target, source_fpfh, target_fpfh,
                                     voxel_size)

    if return_fgr:
        result_fgr = execute_fast_global_registration(source_down, target_down,
                                                      source_fpfh, target_fpfh,
                                                      voxel_size)
        return result_ransac, result_icp, result_fgr, source, target
    return result_ransac, result_icp, source, target


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    # NOTE below is copied from
    # http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Fast-global-registration
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result
