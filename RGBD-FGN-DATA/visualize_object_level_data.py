import os
import glob
import json
from scipy.spatial.transform.rotation import Rotation as sciR
import numpy as np
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
    help="Dataset dir where your object level data exists",
    default="object_level_data",
)
parser.add_argument(
    "--obj_class",
    type=str,
    help="Object class name",
    default="bottle",
)
parser.add_argument(
    "--obj_subclass",
    type=str,
    help="Object subclass name. Be sure to exist in 'obj_class'",
    default="bottle_1",
)
parser.add_argument(
    "--grasp_subsample_rate",
    type=int,
    help="Grasp subsample rate (1 for show all grasps)",
    default=20,
)
parser.add_argument(
    "--show_all",
    type=str2bool,
    help="Shows all meshes sequentially. 'obj_class', 'obj_subclass' are meaningless if show_all=True",
    default=False,
)


def get_o3d_mesh(mesh_path):
    mesh_obj = o3d.io.read_triangle_mesh(
        os.path.join(mesh_path), enable_post_processing=True
    )
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    return mesh_obj, origin_frame


def get_o3d_grasp_infos(
    poses, widths, labels, scores, control_points, subsample_rate=20
):
    ## Parameters
    color_mode = 1  # 1: label mode, 2: score mode

    a1 = 82.7 * np.pi / 180
    d1 = 0.00064
    d2 = 0.0522
    d3 = 0.0186
    d4 = 0.070

    gripper_box_mesh = o3d.geometry.TriangleMesh()

    control_points_partial = np.zeros((0, 3))

    scores_minmax = (min(scores), max(scores))

    # print(scores_minmax_dict)
    idx_arr = np.arange(0, poses.shape[0])
    # np.random.shuffle(idx_arr)

    for i in idx_arr[::subsample_rate]:
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
            if labels[i] == 0:
                color = [0.0, 0.0, 1.0]
            elif labels[i] == 1:
                color = [1.0, 0.0, 0.0]
        elif color_mode == 2:
            if scores_minmax[0] == scores_minmax[1]:
                color = [
                    0.0,
                    0.0,
                    1.0,
                ]
            else:
                min_score = scores_minmax[0]
                max_score = scores_minmax[1]
                alpha = 0.5
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


def main(args):
    dataset_dir = args.dataset_dir
    obj_class = args.obj_class
    obj_subclass = args.obj_subclass
    grasp_subsample_rate = args.grasp_subsample_rate
    show_all = args.show_all
    print("=== arguments ===")
    print("dataset_dir:", dataset_dir)
    print("obj_class:", obj_class)
    print("obj_subclass:", obj_subclass)
    print("grasp_subsample_rate:", grasp_subsample_rate)
    print("show_all:", show_all)

    obj_class_subclass_list = []
    if show_all:
        obj_class_list = [
            name
            for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, name))
        ]
        for obj_class in obj_class_list:
            obj_subclass_list = [
                name
                for name in os.listdir(os.path.join(dataset_dir, obj_class))
                if os.path.isdir(os.path.join(dataset_dir, obj_class, name))
            ]
            for obj_subclass in obj_subclass_list:
                obj_class_subclass_list.append(os.path.join(obj_class, obj_subclass))
    else:
        obj_class_subclass_list.append(os.path.join(obj_class, obj_subclass))

    obj_class_subclass_list.sort()

    for obj_class_subclass in obj_class_subclass_list:
        print()
        print("Loading: ", os.path.join(dataset_dir, obj_class_subclass))

        # Load Mesh
        mesh_path = glob.glob(
            os.path.join(dataset_dir, obj_class_subclass, "*_modified.obj")
        )[0]
        o3d_mesh_obj, o3d_origin_frame = get_o3d_mesh(mesh_path)

        # Load Grasp Samples
        with open(
            os.path.join(dataset_dir, obj_class_subclass, "gsamples.json"), "r"
        ) as f:
            gsamples_json = json.load(f)
            grasp_poses = np.array(gsamples_json["grasp_poses"])
            grip_widths = np.array(gsamples_json["grip_widths"])
            gq_antipodals = np.array(gsamples_json["gq_antipodals"])
            gq_contacts = np.array(gsamples_json["gq_contacts"])
            gq_displacements = np.array(gsamples_json["gq_displacements"])
            grasp_labels = np.array(gsamples_json["grasp_labels"])
            control_points = np.array(gsamples_json["control_points"])
        print("=== grasp samples ===")
        print("grasp_poses :", np.shape(grasp_poses))
        print("grip_widths :", np.shape(grip_widths))
        print("gq_antipodals :", np.shape(gq_antipodals))
        print("gq_contacts :", np.shape(gq_contacts))
        print("gq_displacements :", np.shape(gq_displacements))
        print("grasp_labels :", np.shape(grasp_labels))
        print("control_points :", np.shape(control_points))

        o3d_gripper_box_mesh, o3d_control_points = get_o3d_grasp_infos(
            grasp_poses,
            grip_widths,
            grasp_labels,
            gq_antipodals + gq_contacts + gq_displacements,
            control_points,
            grasp_subsample_rate,
        )

        # Open3D visualization
        o3d_vis = o3d.visualization.Visualizer()
        o3d_vis.create_window()

        render_option = o3d_vis.get_render_option()
        render_option.mesh_show_wireframe = False

        o3d_geometry_list = []
        o3d_geometry_list.append(o3d_mesh_obj)
        o3d_geometry_list.append(o3d_origin_frame)
        o3d_geometry_list.append(o3d_gripper_box_mesh)
        o3d_geometry_list.append(o3d_control_points)
        o3d.visualization.draw_geometries(o3d_geometry_list)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
