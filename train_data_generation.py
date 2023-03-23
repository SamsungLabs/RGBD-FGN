import numpy as np
import glob
import os
from PIL import Image
from scipy.spatial.transform import Rotation as sciR
from scipy.spatial import cKDTree
from common import depth_to_pc_downsampling, find_gt

if __name__ == "__main__":
    data_dir = 'RGBD-FGN-DATA/scene_level_data'
    save_folder = 'train_data'
    frame_interval = 30
    radius = 0.005
    if not os.path.exists(os.path.join(data_dir, save_folder)):
        os.mkdir(os.path.join(data_dir, save_folder))

    scene_paths = sorted(glob.glob(os.path.join(data_dir, '*')))
    start_scene_num = 0
    end_scene_num = len(scene_paths)-1

    grasp_position_distance_max = []
    grasp_position_distance_min = []

    pc_data = [] # (N_data, N_downsize, 3)
    rgb_data = [] # (N_data, 480, 640, 3)
    grasp_approach_direction_gt_wrt_cam_data = [] # (N_data, N_downsize, 3)
    grasp_lateral_direction_gt_wrt_cam_data = [] # (N_data, N_downsize, 3)
    grasp_position_distance_gt_data = [] # (N_data, N_downsize)
    grip_width_gt_data = [] # (N_data, N_downsize)
    grasp_score_gt_data = [] # (N_data, N_downsize)
    grasp_collision_gt_data = [] # (N_data, N_downsize)
    occupancy_gt_data = [] # (N_data, N_downsize)

    for i in range(start_scene_num, end_scene_num):
        grasp_paths = glob.glob(os.path.join(scene_paths[i], 'scene_grasp_infos', '*'))
        scene_grasp_data = np.load(grasp_paths[0])

        grasp_poses = scene_grasp_data["grasp_poses"].astype(np.float32) # (N, 7) 
        grasp_position = grasp_poses[:,:3] # (N, 3)
        grasp_R = sciR.from_quat(grasp_poses[:,3:]).as_matrix() # (N, 3, 3)
        grasp_approach_direction = grasp_R[:,:,2] # (N, 3)
        grasp_lateral_direction = grasp_R[:,:,1] # (N, 3)
        grasp_upper_direction = -grasp_R[:,:,0] # (N, 3)
        
        control_points = scene_grasp_data["grasp_control_points"].astype(np.float32) # (N, 3)
        grasp_collision = scene_grasp_data["grasp_collisions"].astype(np.float32) # (N, )

        grasp_quality = scene_grasp_data["grasp_quality_antipodals"] + \
                            scene_grasp_data["grasp_quality_contacts"] + \
                            scene_grasp_data["grasp_quality_displacements"] # (N, )

        for k in range(grasp_position.shape[0]):
            dist = np.dot(grasp_position[k,:] - control_points[k,:], grasp_R[k,:,2])
            grasp_position[k,:] = control_points[k,:] + dist * grasp_R[k,:,2]

        grasp_position_distance = np.sum(np.multiply(grasp_position - control_points, grasp_approach_direction), axis=1) # (N, )
        grasp_position_distance = np.maximum(grasp_position_distance, 0.) # (N, )
        grasp_position_distance_max.append(np.max(grasp_position_distance))
        grasp_position_distance_min.append(np.min(grasp_position_distance))
    
        grip_width = scene_grasp_data['grasp_widths'].astype(np.float32) # (N, )
        grip_width = np.maximum(grip_width, 0.115/10) # (N, ), gripper maximum distance = 0.115

        rgb_paths = sorted(glob.glob(os.path.join(scene_paths[i], 'rgb', '*')))
        depth_paths = sorted(glob.glob(os.path.join(scene_paths[i], 'depth', '*')))
        meta_paths = sorted(glob.glob(os.path.join(scene_paths[i], 'meta', '*')))
        seg_paths = sorted(glob.glob(os.path.join(scene_paths[i], 'seg_id', '*')))
        syn_depth_paths = sorted(glob.glob(os.path.join(scene_paths[i], 'depth_syn', '*')))

        _tree = cKDTree(control_points)

        for j in range(100, len(depth_paths), frame_interval):
            rgb = np.array(Image.open(rgb_paths[j]))
            depth = np.array(Image.open(depth_paths[j]), dtype=np.float32) / 1000.0
            meta = np.load(meta_paths[j])
            seg = np.array(Image.open(seg_paths[j]))
            syn_depth = np.array(Image.open(syn_depth_paths[j]), dtype=np.float32) / 1000.0

            cam_pose = np.eye(4)
            cam_pose[:3, 3] = meta['cam_pose'][:3]
            cam_pose[:3, :3] = sciR.from_quat(meta['cam_pose'][3:]).as_matrix()

            pc, _, occupancy = depth_to_pc_downsampling(depth, meta, rgb, seg)

            pc_wo_noise, _, _ = depth_to_pc_downsampling(syn_depth, meta, rgb, seg)
            grasp_approach_direction_wrt_cam, grasp_lateral_direction_wrt_cam, grasp_position_distance_gt, grip_width_gt, grasp_score_gt, grasp_collision_gt = find_gt(
                _tree, pc_wo_noise, grasp_approach_direction, grasp_lateral_direction, grasp_upper_direction, grasp_position_distance, grip_width, grasp_collision, grasp_quality, cam_pose, radius)

            pc_data.append(pc)
            rgb_data.append(rgb)
            grasp_approach_direction_gt_wrt_cam_data.append(grasp_approach_direction_wrt_cam)
            grasp_lateral_direction_gt_wrt_cam_data.append(grasp_lateral_direction_wrt_cam)
            grasp_position_distance_gt_data.append(grasp_position_distance_gt)
            grip_width_gt_data.append(grip_width_gt)
            grasp_score_gt_data.append(grasp_score_gt)
            grasp_collision_gt_data.append(grasp_collision_gt)
            occupancy_gt_data.append(occupancy)
            
        print("scene num: ", i)

    print(np.max(np.array(grasp_position_distance_max)))
    print(np.min(np.array(grasp_position_distance_min)))

    np.savez_compressed(os.path.join(data_dir, save_folder, 'output_data.npz'),
        grasp_approach_direction_gt_wrt_cam=grasp_approach_direction_gt_wrt_cam_data,
        grasp_lateral_direction_gt_wrt_cam=grasp_lateral_direction_gt_wrt_cam_data,
        grasp_position_distance_gt=grasp_position_distance_gt_data,
        grip_width_gt=grip_width_gt_data,
        grasp_score_gt=grasp_score_gt_data,
        grasp_collision_gt=grasp_collision_gt_data,
        occupancy_gt=occupancy_gt_data
    )

    np.savez_compressed(os.path.join(data_dir, save_folder, 'position_distance_minmax.npz'),
        grasp_position_distance_max = np.max(np.array(grasp_position_distance_max)),
        grasp_position_distance_min = np.min(np.array(grasp_position_distance_min))
    )
    
    pc_data = np.array(pc_data)
    rgb_data = np.array(rgb_data)

    np.savez_compressed(os.path.join(data_dir, save_folder, 'input_data.npz'),
        pc=pc_data,
        rgb=rgb_data
    )    
    
