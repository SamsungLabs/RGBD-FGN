import numpy as np
import open3d as o3d

class Visualizer():
	def __init__(self):
		self._vis = o3d.visualization.Visualizer()
		self._vis.create_window('Visualizer', width=640, height=480)
		self._vis_ini = True
		self._pcd = o3d.geometry.PointCloud()
		self._line = o3d.geometry.LineSet()
		self._thick_line_mesh = o3d.geometry.TriangleMesh()

	def update_scene(self, pc, rgb, grasp_poses, grip_widths, grasp_score=None):
		# pc: np array (n, 3) # x,y,z
		# rgb: np array (n, 3) # r,g,b
		# grasp_poses: np array (N, 4, 4) w.r.t global frame
		# grip_widths: np array (N, )
		# grasp_score: np array (N, )

		# Set pointcloud
		self._pcd.points = o3d.utility.Vector3dVector(pc)
		self._pcd.colors = o3d.utility.Vector3dVector(rgb/255.0)

		# Set gripper
		line_count = 0
		p_gripper_global_tot = np.empty((0, 3))
		lines = np.empty((0, 2))
		colors = np.empty((0, 3))

		d1 = 0.64/1000.0
		d2 = 52.2/1000.0
		d4 = 70.0/1000.0
		a1 = 82.7*np.pi/180.0
		for i in range(grasp_poses.shape[0]):
			del_w = grip_widths[i]
			del_a1 = np.arccos(-del_w/(2.0*d4) + np.cos(a1)) - a1
			del_l = d4 * (np.sin(a1) - np.sin(a1+del_a1))
			half_th = 0.003
			p_gripper_body = np.array([[-half_th, (d1+del_w)/2.0+half_th, -del_l],
										[-half_th, (d1+del_w)/2.0-half_th, -del_l],
										[-half_th, (d1+del_w)/2.0+half_th, -d2-del_l-half_th],
										[-half_th, (d1+del_w)/2.0-half_th, -d2-del_l],
										[-half_th, -(d1+del_w)/2.0-half_th, -d2-del_l-half_th],
										[-half_th, -(d1+del_w)/2.0+half_th, -d2-del_l],
										[-half_th, -(d1+del_w)/2.0-half_th, -del_l],
										[-half_th, -(d1+del_w)/2.0+half_th, -del_l],
										[half_th, (d1+del_w)/2.0+half_th, -del_l],
										[half_th, (d1+del_w)/2.0-half_th, -del_l],
										[half_th, (d1+del_w)/2.0+half_th, -d2-del_l-half_th],
										[half_th, (d1+del_w)/2.0-half_th, -d2-del_l],
										[half_th, -(d1+del_w)/2.0-half_th, -d2-del_l-half_th],
										[half_th, -(d1+del_w)/2.0+half_th, -d2-del_l],
										[half_th, -(d1+del_w)/2.0-half_th, -del_l],
										[half_th, -(d1+del_w)/2.0+half_th, -del_l],])
			p_gripper_global = grasp_poses[i,:3,3] + p_gripper_body @ np.transpose(grasp_poses[i,:3,:3])
			p_gripper_global_tot = np.vstack((p_gripper_global_tot, p_gripper_global))
			lines = np.vstack((lines, np.array([[16*line_count, 16*line_count+1],
												[16*line_count, 16*line_count+2],
												[16*line_count+1, 16*line_count+3],
												[16*line_count+3, 16*line_count+5],												
												[16*line_count+2, 16*line_count+4],
												[16*line_count+4, 16*line_count+6],
												[16*line_count+5, 16*line_count+7],
												[16*line_count+6, 16*line_count+7],
												[16*line_count, 16*line_count+8],
												[16*line_count+1, 16*line_count+9],
												[16*line_count+2, 16*line_count+10],
												[16*line_count+3, 16*line_count+11],
												[16*line_count+4, 16*line_count+12],
												[16*line_count+5, 16*line_count+13],
												[16*line_count+6, 16*line_count+14],
												[16*line_count+7, 16*line_count+15],
												[16*line_count+8, 16*line_count+9],
												[16*line_count+8, 16*line_count+10],
												[16*line_count+9, 16*line_count+11],
												[16*line_count+11, 16*line_count+13],												
												[16*line_count+10, 16*line_count+12],
												[16*line_count+12, 16*line_count+14],
												[16*line_count+13, 16*line_count+15],
												[16*line_count+14, 16*line_count+15]])))
			if grasp_score is None:
				colors = np.vstack((colors, np.tile(np.array([1., 1., 1.]),[24,1])))
			else:
				color_tmp = (grasp_score[i] - np.min(grasp_score)) / (np.max(grasp_score) - np.min(grasp_score))
				colors = np.vstack((colors, np.tile(np.array([color_tmp, 0., 0.]),[24,1])))

			line_count += 1

		self._line.points = o3d.utility.Vector3dVector(p_gripper_global_tot)
		self._line.lines = o3d.utility.Vector2iVector(lines)
		self._line.colors = o3d.utility.Vector3dVector(colors)		

		# Draw
		if self._vis_ini:
			self._vis.add_geometry(self._pcd,reset_bounding_box=True)
			self._vis.add_geometry(self._line, reset_bounding_box=False)
			self._vis_ini = False
			opt = self._vis.get_render_option()
			opt.background_color = [0, 0, 0]
			opt.light_on = True

		self._vis.update_geometry(self._pcd)
		self._vis.update_geometry(self._line)

	def update_render(self):
		self._vis.poll_events()
		self._vis.update_renderer()
