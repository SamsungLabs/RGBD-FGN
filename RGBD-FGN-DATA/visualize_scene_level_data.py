import os
import glob
import copy
import random
from scipy.spatial.transform.rotation import Rotation as sciR
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import open3d as o3d
from parse import *
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Dataset dir where your scene level data exists",
    default="scene_level_data",
)
parser.add_argument(
    "--scene_num",
    type=str,
    help="Number of scene in 8 digits",
    default="00000000",
)
parser.add_argument(
    "--frame_num", type=str, help="Number of frame in 8 digits", default="00000000"
)
parser.add_argument(
    "--grasp_subsample_rate",
    type=int,
    help="Grasp subsample rate (1 for show all grasps)",
    default=2,
)
parser.add_argument(
    "--show_all_pcd",
    type=str2bool,
    help="Shows point cloud reconstructed from all view of frames",
    default=True,
)
parser.add_argument(
    "--show_masked_rgb",
    type=str2bool,
    help="Shows RGB image masked using segmentation",
    default=True,
)

parser.add_argument("--show_pc", type=str2bool, help="Show Point Cloud", default=True)


def get_o3d_grasp_infos(
    poses, widths, scores, collisions, control_points, seg_ids, subsample_rate
):
    ## Parameters
    color_mode = 2  # 1: collision show mode, 2: score mode

    a1 = 82.7 * np.pi / 180
    d1 = 0.00064
    d2 = 0.0522
    d3 = 0.0186
    d4 = 0.070

    gripper_box_mesh = o3d.geometry.TriangleMesh()

    control_points_partial = np.zeros((0, 3))

    scores_dict = {}
    scores_minmax_dict = {}
    if color_mode == 2:
        for i in range(0, scores.shape[0]):
            if collisions[i] == True:
                continue
            if seg_ids[i] in scores_dict.keys():
                scores_dict[seg_ids[i]].append(scores[i])
            else:
                scores_dict[seg_ids[i]] = [scores[i]]

        for key in scores_dict.keys():
            scores_minmax_dict[key] = (min(scores_dict[key]), max(scores_dict[key]))

    # print(scores_minmax_dict)
    idx_arr = np.arange(0, poses.shape[0])
    # np.random.shuffle(idx_arr)

    for i in idx_arr[::subsample_rate]:
        if color_mode == 2 and collisions[i] == True:
            continue

        # Add points
        del_w = widths[i]
        del_a1 = np.arccos(-del_w / (2.0 * d4) + np.cos(a1)) - a1
        # del_l = d4 * (np.sin(a1) - np.sin(a1 + del_a1))
        del_l = 0

        line_width = 0.003
        box_origin_body = np.array(
            [
                [
                    -line_width / 2,
                    (d1 + del_w) / 2.0,
                    -d2 - del_l - line_width,
                ],
                [
                    -line_width / 2,
                    -(d1 + del_w) / 2.0,
                    -d2 - del_l - line_width,
                ],
                [
                    -line_width / 2,
                    -(d1 + del_w) / 2.0 - line_width,
                    -d2 - del_l - line_width,
                ],
            ]
        )
        box_origin_global = poses[i, :3] + box_origin_body @ np.transpose(
            sciR.from_quat(poses[i, 3:]).as_matrix()
        )

        # Add colors
        if color_mode == 1:
            color = [1.0, 1.0, 1.0]
            if collisions[i] == True:
                color = [0.0, 0.0, 0.0]
            else:
                color = [1.0, 0.0, 0.0]
        elif color_mode == 2:
            if scores_minmax_dict[seg_ids[i]][0] == scores_minmax_dict[seg_ids[i]][1]:
                color = [
                    0.0,
                    0.0,
                    1.0,
                ]
            else:
                min_score = scores_minmax_dict[seg_ids[i]][0]
                max_score = scores_minmax_dict[seg_ids[i]][1]
                alpha = 0.0
                thresh_score = max_score * alpha + min_score * (1 - alpha)
                if scores[i] >= thresh_score:
                    color = [
                        1.0,
                        1.0,
                        1.0,
                    ]
                else:
                    color = [
                        0.0,
                        0.0,
                        0.0,
                    ]

        for j in range(3):
            T_wj = np.eye(4)
            T_wj[:3, :3] = sciR.from_quat(poses[i, 3:]).as_matrix()

            if j == 0:
                x_len = line_width
                y_len = line_width
                z_len = d2 + line_width
                T_wj[:3, 3] = box_origin_global[j, :]
            elif j == 1:
                x_len = line_width
                y_len = d1 + del_w
                z_len = line_width
                T_wj[:3, 3] = box_origin_global[j, :]
            elif j == 2:
                x_len = line_width
                y_len = line_width
                z_len = d2 + line_width
                T_wj[:3, 3] = box_origin_global[j, :]

            box = o3d.geometry.TriangleMesh.create_box(
                width=x_len, height=y_len, depth=z_len
            )
            box.compute_vertex_normals()
            box.transform(T_wj)

            box.paint_uniform_color(color)
            gripper_box_mesh += box

        # Add control points
        control_points_partial = np.vstack(
            (control_points_partial, control_points[i, :])
        )

    cp_len = np.shape(control_points_partial)[0]
    pcd_control_points = o3d.geometry.PointCloud()
    pcd_control_points.points = o3d.utility.Vector3dVector(control_points_partial)
    pcd_control_points.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([1.0, 1.0, 0.0]), (cp_len, 1))
    )

    return gripper_box_mesh, pcd_control_points


def calc_pc_xyz_rgb(intrinsic, T_0_cam, rgb, depth, downsize=False):
    """
    input :
    - intrinsic: camera intrinsic = [W,H,fx,fy,cx,cy]
    - T_0_cam : 4x4 SE3
    - rgb : height x width x 3  rgb image (unit : 0~255)
    - depth : height x width (unit : mm)

    output :
    - pc_xyz : N x 3 (unit : m)
    - pc_rgb : N x 3 (unit : 0~255)
    """
    _, _, fx, fy, cx, cy = intrinsic

    depth_m = copy.deepcopy(depth).squeeze() / 1000.0
    mask = np.where((depth_m > 0.001) & (depth_m < 3.0))

    if len(mask[0]) == 0 or len(mask[0]) != len(mask[1]):
        return

    x, y = mask[1], mask[0]

    if downsize == True:
        downsize_rate = 20
        rand_idx_list = random.choices(range(len(x)), k=int(len(x) / downsize_rate))
        x = x[rand_idx_list]
        y = y[rand_idx_list]

    normalized_x = x.astype(np.float32) - cx
    normalized_y = y.astype(np.float32) - cy

    pc_x = normalized_x * depth_m[y, x] / fx
    pc_y = normalized_y * depth_m[y, x] / fy
    pc_z = depth_m[y, x]
    pc_xyz = np.vstack((pc_x, pc_y, pc_z)).T
    pc_xyz = np.matmul(T_0_cam, np.vstack((pc_x, pc_y, pc_z, np.ones_like(pc_x))))
    pc_xyz = pc_xyz[:3, :].T
    pc_rgb = rgb[y, x, :]

    return pc_xyz, pc_rgb


def p_quat_from_T(T):
    p = T[0:3, 3]
    quat = sciR.from_matrix(T[0:3, 0:3]).as_quat()
    return p, quat


def main(args):
    dataset_dir = args.dataset_dir
    scene_num = args.scene_num
    frame_num = args.frame_num
    show_pc = args.show_pc
    grasp_subsample_rate = args.grasp_subsample_rate
    show_all_pcd = args.show_all_pcd
    show_masked_rgb = args.show_masked_rgb

    print("=== arguments ===")
    print("dataset_dir :", dataset_dir)
    print("scene_num :", scene_num)
    print("frame_num :", frame_num)
    print("grasp_subsample_rate :", grasp_subsample_rate)
    print("show_all_pcd :", show_all_pcd)
    print("show_masked_rgb :", show_masked_rgb)
    print("show_pc :", show_pc)

    scene_dir = "scene_" + scene_num
    file_name = scene_num + "_" + frame_num
    ##################
    #### Decoder #####
    ##################
    rgb_img = tf.io.decode_jpeg(
        contents=tf.io.read_file(
            os.path.join(dataset_dir, scene_dir, "rgb", file_name + ".jpeg")
        ),
        channels=3,
    )
    print("=== rgb ===")
    print("rgb :", np.shape(rgb_img))

    depth = tf.io.decode_png(
        contents=tf.io.read_file(
            os.path.join(dataset_dir, scene_dir, "depth", file_name + ".png")
        ),
        channels=1,
        dtype=tf.dtypes.uint16,
    )
    print("=== depth ===")
    print("depth :", np.shape(depth))

    depth_syn = depth
    if os.path.exists(os.path.join(dataset_dir, scene_dir, "depth_syn")):
        depth_syn = tf.io.decode_png(
            contents=tf.io.read_file(
                os.path.join(dataset_dir, scene_dir, "depth_syn", file_name + ".png")
            ),
            channels=1,
            dtype=tf.dtypes.uint16,
        )
    print("=== depth_syn ===")
    print("depth_syn :", np.shape(depth_syn))

    seg_id = tf.io.decode_png(
        contents=tf.io.read_file(
            os.path.join(dataset_dir, scene_dir, "seg_id", file_name + ".png")
        ),
        channels=1,
        dtype=tf.dtypes.uint8,
    )
    print("=== seg_id ===")
    print("seg_id :", np.shape(seg_id))

    label_id = seg_id
    if os.path.exists(os.path.join(dataset_dir, scene_dir, "label_id")):
        label_id = tf.io.decode_png(
            contents=tf.io.read_file(
                os.path.join(dataset_dir, scene_dir, "label_id", file_name + ".png")
            ),
            channels=1,
            dtype=tf.dtypes.uint8,
        )
    print("=== label_id ===")
    print("label_id :", np.shape(label_id))

    meta = np.load(os.path.join(dataset_dir, scene_dir, "meta", file_name + ".npz"))
    cam_pose = meta["cam_pose"]
    intrinsic = meta["intrinsic"]
    table_seg_id = meta["table_seg_id"]
    print("=== meta ===")
    print("cam_pose :", np.shape(cam_pose))
    print("intrinsic :", np.shape(intrinsic))
    print("table_seg_id :", np.shape(table_seg_id))

    scene_grasp_infos = np.load(
        os.path.join(
            dataset_dir,
            scene_dir,
            "scene_grasp_infos",
            "scene_" + scene_num + ".npz",
        )
    )
    grasp_poses = scene_grasp_infos["grasp_poses"]
    grasp_widths = scene_grasp_infos["grasp_widths"]
    grasp_quality_antipodals = scene_grasp_infos["grasp_quality_antipodals"]
    grasp_quality_contacts = scene_grasp_infos["grasp_quality_contacts"]
    grasp_quality_displacements = scene_grasp_infos["grasp_quality_displacements"]
    grasp_collisions = scene_grasp_infos["grasp_collisions"]
    grasp_seg_ids = scene_grasp_infos["grasp_seg_ids"]
    grasp_control_points = scene_grasp_infos["grasp_control_points"]
    print("=== scene_grasp_infos ===")
    print("grasp_poses :", np.shape(grasp_poses))
    print("grasp_widths :", np.shape(grasp_widths))
    print("grasp_quality_antipodals :", np.shape(grasp_quality_antipodals))
    print("grasp_quality_contacts :", np.shape(grasp_quality_contacts))
    print("grasp_quality_displacements :", np.shape(grasp_quality_displacements))
    print("grasp_collisions :", np.shape(grasp_collisions))
    print("grasp_seg_ids :", np.shape(grasp_seg_ids))
    print("grasp_control_points : ", np.shape(grasp_control_points))

    ##############################
    #### Visualize [Open3d] ####
    ##############################
    if show_pc:
        o3d_geometry_list = []

        pcd = o3d.geometry.PointCloud()
        pc_xyz = np.zeros((0, 3))
        pc_rgb = np.zeros((0, 3))
        if show_all_pcd:
            # Visualize point cloud with all views
            rgb_file_list = glob.glob(
                os.path.join(dataset_dir, scene_dir, "rgb", "*.jpeg")
            )
            rgb_file_list.sort()

            depth_file_list = glob.glob(
                os.path.join(dataset_dir, scene_dir, "depth", "*.png")
            )
            depth_file_list.sort()

            meta_file_list = glob.glob(
                os.path.join(dataset_dir, scene_dir, "meta", "*.npz")
            )
            meta_file_list.sort()

            for i in range(0, len(rgb_file_list), 20):
                rgb_img_i = tf.io.decode_jpeg(
                    contents=tf.io.read_file(rgb_file_list[i]),
                    channels=3,
                )

                depth_img_i = tf.io.decode_png(
                    contents=tf.io.read_file(depth_file_list[i]),
                    channels=1,
                    dtype=tf.dtypes.uint16,
                )
                meta_i = np.load(os.path.join(meta_file_list[i]))
                cam_pose_i = meta_i["cam_pose"]
                intrinsic_i = meta_i["intrinsic"]

                cam_pose_SE3_i = np.eye(4)
                cam_pose_SE3_i[:3, 3] = cam_pose_i[:3]
                cam_pose_SE3_i[:3, :3] = sciR.from_quat(cam_pose_i[3:]).as_matrix()
                pc_xyz_i, pc_rgb_i = calc_pc_xyz_rgb(
                    intrinsic_i,
                    cam_pose_SE3_i,
                    rgb_img_i.numpy(),
                    depth_img_i.numpy(),
                    downsize=True,
                )
                pc_xyz = np.vstack((pc_xyz, pc_xyz_i))
                pc_rgb = np.vstack((pc_rgb, pc_rgb_i))

        else:
            # Visualize point cloud obtained from current camera view
            cam_pose_SE3 = np.eye(4)
            cam_pose_SE3[:3, 3] = cam_pose[:3]
            cam_pose_SE3[:3, :3] = sciR.from_quat(cam_pose[3:]).as_matrix()
            pc_xyz, pc_rgb = calc_pc_xyz_rgb(
                intrinsic, cam_pose_SE3, rgb_img.numpy(), depth.numpy(), downsize=False
            )

        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)
        o3d_geometry_list.append(pcd)

        gripper_box_mesh, pcd_control_points = get_o3d_grasp_infos(
            grasp_poses,
            grasp_widths,
            grasp_quality_antipodals
            + grasp_quality_displacements
            + grasp_quality_contacts,
            grasp_collisions,
            grasp_control_points,
            grasp_seg_ids,
            grasp_subsample_rate,
        )
        o3d_geometry_list.append(gripper_box_mesh)
        o3d_geometry_list.append(pcd_control_points)

        o3d.visualization.draw_geometries(o3d_geometry_list)

    ############################
    #### Visualize [Image] #####
    ############################
    if show_masked_rgb:
        mask = seg_id.numpy() > 0
        mask = mask.astype(int)
        mask = np.concatenate(
            (mask, np.zeros((np.shape(mask)[0], np.shape(mask)[1], 2))), axis=-1
        )
        rgb_img = rgb_img + 100 * mask
    plt.figure(figsize=[40, 20])
    plt.subplot(1, 5, 1)
    plt.imshow(rgb_img.numpy())
    if show_masked_rgb:
        plt.title("RGB (Masked w/ Segmentation)")
    else:
        plt.title("RGB")
    plt.subplot(1, 5, 2)
    plt.imshow(depth.numpy(), cmap="prism")
    plt.title("Depth")
    plt.subplot(1, 5, 3)
    plt.imshow(depth_syn.numpy(), cmap="prism")
    plt.title("Depth Synthetic")
    plt.subplot(1, 5, 4)
    plt.imshow(seg_id.numpy(), cmap="flag")
    plt.title("Segmentation ID")
    plt.subplot(1, 5, 5)
    plt.imshow(label_id.numpy(), cmap="flag")
    plt.title("Label ID")
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
