#!/usr/bin/python
# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
import rospy
from geometry_msgs.msg import PointStamped, Point
import cv2
from easydict import EasyDict as edict
import argparse
import numpy as np
from numpy.linalg import norm
from copy import copy
import matplotlib.pyplot as plt
import torch
from math import pi
from queue import Queue
import tf
from tf.transformations import euler_from_quaternion
from scene_understand.perceptor import Perceptor
from utils.utils import read_yaml, mid_perpendicular_lines, get_new_pose, pixel_to_world, _normalize_heading, neighbor_bfs_free, _world_to_map2, _map_to_world, compute_normal
from utils.realsense import RealSenseCamera, RealSenseCameraDepth
from utils.ros import ROS
from arguments import get_args

FREE = 0
OBSTACLE = 100
UNKNOWN = -1

class Viewpoint:
    def __init__(self, cfg, ros_cfg, perceptor, ros):
        self.viewpoint_topic = ros_cfg.viewpoint_topic
        self.viewpoint_score_topic = ros_cfg.viewpoint_score_topic
        self.viewpoint_finished_topic = ros_cfg.finished_viewpoint_topic
        self.detect_visited = []
        self.cur_finished_point = []
        self.prev_finished_idx = -1
        self.cur_finished_idx = -1
        self.detect_results = []
        self.detect_robot_poses = []
        self.detect_text_poses = []
        self.detect_viewpoints = []
        
        self.map_frame = ros_cfg.frames.map
        self.base_frame = ros_cfg.frames.base
        self.afford_ratio = ros_cfg.afford_ratio
        self.camera_id = ros_cfg.camera_id
        self.frame_rate = ros_cfg.frame_rate
        self.goal_frame = ros_cfg.frames.goal
        self.min_move = ros_cfg.min_move
        self.min_yaw_change = eval(ros_cfg.min_yaw_change)
        self.decimal = ros_cfg.decimal
        
        self.perceptor = perceptor
        self.ros = ros
        self.viewpoint_pub = rospy.Publisher(self.viewpoint_topic, PointStamped, queue_size=10)
        self.viewpoint_score_pub = rospy.Publisher(self.viewpoint_score_topic, Point, queue_size=10)
        rospy.Subscriber(self.viewpoint_finished_topic, PointStamped, self.finished_callback)
        
    
    def finished_callback(self, msg):
        finished_point = [msg.point.x, msg.point.y, msg.point.z]
        print('get finished point:', finished_point)
    

    
    def generate_viewpoint(self, depth, bbox, cur_pose_world, center_map, show=False):
        xl, yl, xr, yr = bbox.astype(np.int64)
        xl = min(xl, depth.shape[1] - 1)
        yl = min(yl, depth.shape[0] - 1)
        xr = min(xr, depth.shape[1] - 1)
        yr = min(yr, depth.shape[0] - 1)
        if center_map is not None:
            ## get viewpoint for active vision (in world coord: SLAM 2D-coord)
            y_mid = int(np.floor((yr + yl) / 2))
            try:
                depth1 = depth[y_mid, int(xl)]
                depth2 = depth[y_mid, int(xr)]
            except Exception as e:
                print('error:', e, xl, xr, y_mid)
            local1 = self.ros._pixel_to_local([int(xl), y_mid], depth1)  # x, y, z -- SLAM coord
            local2 = self.ros._pixel_to_local([int(xr), y_mid], depth2)
            height = (local1[2] + local2[2]) / 2
    
            #1) perpendicular line (works better)
            mid_point_local = np.array([(local1[0] + local2[0]) / 2, (local1[1] + local2[1]) / 2])
            A, B, C = mid_perpendicular_lines(local1[0], local1[1], local2[0], local2[1]) 
            if B == 0:
                theta = pi / 2 if -A > 0 else -pi / 2
            else:
                theta = np.arctan(-A / B)
            
            #2) normal vector
            # left_top = np.array(self.ros._pixel_to_local([xl, yl], depth[yl, xl]))
            # right_top = np.array(self.ros._pixel_to_local([xr, yl], depth[yl, xr]))
            # left_down = np.array(self.ros._pixel_to_local([xl, yr], depth[yr, xl]))
            # right_down = np.array(self.ros._pixel_to_local([xr, yr], depth[yr, xr]))
            # right_mid = (right_top + right_down) / 2
            # left_mid = (left_top + left_down) / 2
            # vector_horizon = left_mid - right_mid
            # top_mid = (right_top + left_top) / 2
            # down_mid = (right_down + left_down) / 2
            # vector_vertical = top_mid - down_mid 
            # body_normal = compute_normal(vector_horizon, vector_vertical)
            # theta = _normalize_heading(np.arctan2(body_normal[1], body_normal[0]))  # body orientation
            
            dxy = (self.afford_ratio * height) * np.array([np.cos(theta), np.sin(theta)]) 
            # or replace (self.afford_ratio * height) with a predefined distance
            
            viewpoint_local = mid_point_local - dxy
            viewpoint_local = np.append(viewpoint_local, theta)  # [x, y, theta]
            viewpoint_world = get_new_pose(cur_pose_world, viewpoint_local) # [x, y, cur_pose[2] + theta]
            
            ## world to map
            mapData = copy(self.ros.mapData)
            viewpoint_map = _world_to_map2(mapData, viewpoint_world[:2])  # (my, mx)
            
            if viewpoint_map is None:
                return None, None
            ## search neighbor free point
            map_array = copy(self.ros.map_array)
            if viewpoint_map[0] >= map_array.shape[0] or viewpoint_map[1] >= map_array.shape[1] or \
                map_array[viewpoint_map[0], viewpoint_map[1]] == OBSTACLE:
                
                viewpoint_map = neighbor_bfs_free(map_array, viewpoint_map)  # (mx, my)
                if viewpoint_map is None:
                    return None, None
                viewpoint_world = _map_to_world(mapData, viewpoint_map)
                viewpoint_world = np.append(viewpoint_world, _normalize_heading(cur_pose_world[2] + theta))

            if show: ## plot map
                cur_pose_map = _world_to_map2(mapData, cur_pose_world[:2])
                map_array[np.where(map_array == 0)] = 254 
                map_array[np.where(map_array == 100)] = 0
                map_array[np.where(map_array == -1)] = 205
                cv2.imwrite(f"./nav/raw_map.png", map_array)
                
                height = map_array.shape[0]
                obstacles = np.where(map_array == 0)
                unknown = np.where(map_array == 205)
                plt.scatter(obstacles[1], obstacles[0], color='black')
                plt.scatter(unknown[1], unknown[0], color='gray')
                plt.scatter(center_map[0], center_map[1], color='r')
                plt.scatter(viewpoint_map[1], viewpoint_map[0], color='g')
                plt.scatter(cur_pose_map[1], cur_pose_map[0], color='b')
                plt.savefig('./nav/viewpoints2.png')
                plt.clf()
        else:
            viewpoint_map = None
        return viewpoint_map, np.round(viewpoint_world, self.decimal)
    
    
    def merge_idx(self, cur_pose, existing_poses):
        for idx, exist_pose in enumerate(existing_poses):
            if norm((exist_pose[:2]) - np.array(cur_pose[:2])) <= self.min_move:
                print(f'viewpoint merge to {idx}')
                return idx
        return None

    def _warp_msg(self, point_world):
        point_msg = PointStamped()
        point_msg.header.frame_id = self.goal_frame
        point_msg.header.stamp = rospy.Time().now()
        point_msg.point.x = point_world[0]
        point_msg.point.y = point_world[1]
        point_msg.point.z = point_world[2]
        return point_msg

    def pub_viewpoints(self, viewpoints_world, viewpoints_score):
        for viewpoint, score in zip(viewpoints_world, viewpoints_score):
            score_msg = Point()
            score_msg.x = score
            self.viewpoint_score_pub.publish(score_msg)
            viewpoint_msg = self._warp_msg(viewpoint)
            self.viewpoint_pub.publish(viewpoint_msg)
    
    def update(self, new_instances, centers_world, center_map, distances, 
                    depth_img, depth_colormap, cur_pose, vis=False):
        ## scene text spotting and fusion
        tmp_viewpoints = []
        tmp_text_poses = []
        tmp_res = []
        viewpoints_map = []
        viewpoints_world = []
        for instance, center_world, distance in zip(new_instances, centers_world, distances): 
            if vis:
                xl, yl, xr, yr = instance['bbox'].astype(np.int64)
                cv2.rectangle(depth_colormap, (xl, yl), (xr, yr), (255, 0, 0), 2)  # red color
            
            ## project viewpoint to world and map(ogm) coordinate
            viewpoint_map, viewpoint_world = \
                self.generate_viewpoint(depth_img, instance['bbox'], cur_pose, center_map, show=False)
            if viewpoint_map is None:
                continue
            viewpoint_world = viewpoint_world.tolist()
            if  self.merge_idx(center_world, self.detect_text_poses) is None:  #! bug
                ## 3D semantic fusion
                tmp_viewpoints.append(viewpoint_world)
                tmp_text_poses.append(center_world)
                tmp_res.append({
                    'text_pose': center_world,
                    'text': instance['text'],
                    'prob': instance['prob'],
                    'viewpoint_map': viewpoint_map,
                    'viewpoint_world': viewpoint_world,
                })
                viewpoints_map.append(list(viewpoint_map))
                viewpoints_world.append(viewpoint_world)
                self.detect_results.append(copy(tmp_res[-1]))
                self.detect_visited.append(0)
            
            self.detect_robot_poses.append(copy(cur_pose))
            self.detect_text_poses.append(copy(center_world))  # used for merge judgement
            self.detect_viewpoints.append(copy(viewpoint_world))
            
            
        if len(tmp_viewpoints):
            has_item = True
        else:
            has_item = False
        
        return np.array(viewpoints_map), viewpoints_world, has_item

    
    @torch.no_grad()
    def run(self, plot=False, save_res=False, encode=True):
        step = 1
        prev_pose = self.ros.get_tf()
        has_item = True
        while not rospy.is_shutdown():
            color_img, depth_img, depth_colormap = self.ros.get_image()
            cur_pose = self.ros.get_tf()
            if (norm((cur_pose - prev_pose)[:2]) > self.min_move or \
                abs(cur_pose[2] - prev_pose[2]) > self.min_yaw_change) or has_item or \
                len(self.cur_finished_point) or \
                step % self.frame_rate == 0:  #! only for test
                print("step:", step)
                ## get scene text spotting result
                out = self.perceptor(color_img, plot=plot, save_res=save_res, encode=encode)
                mapData = copy(self.ros.mapData)
                new_instances = []
                new_centers_world = []
                new_centers_map = []
                new_distances = []
                for item in out:
                    center_map, center_world, distance = \
                        self.ros.get_region_depth(depth_img, item, cur_pose, mapData)
                    merge_idx = self.merge_idx(center_world)
                    if not merge_idx or merge_idx != self.cur_finished_idx:
                        new_instances.append(item)
                        new_centers_world.append(center_world)
                        new_centers_map.append(center_map)
                        new_distances.append(distance)
                        
                viewpoints_map, viewpoints_world, has_item = \
                    self.update(new_instances, new_centers_world, new_centers_map, new_distances, depth_img, 
                                depth_colormap, cur_pose, vis=False)
                if not len(self.cur_finished_point):
                    self.pub_viewpoints(viewpoints_world)
                if  len(out):
                    prev_pose = cur_pose
            step += 1
            cv2.imshow(self.camera_id, depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.ros.rate.sleep()
        cv2.destroyAllWindows()
        self.ros.close_camera()


if __name__ == '__main__':
    rospy.init_node('viewpoint_manager')
    args = get_args()
    if args.scene == '1':
        config_path = './config/scene1.yaml'
    elif args.scene == '2':
        config_path = './config/scene2.yaml'
    else:
        raise ValueError(args.scene)
    print('scene:', args.scene)
    cfg = edict(read_yaml(args.config_path))
    cfg.debug = args.debug
    ros_cfg = edict(read_yaml(args.ros_config_path))
    ros_cfg.update(vars(args))
    perceptor = Perceptor(cfg)
    ros = ROS(ros_cfg, args.debug)
    node = Viewpoint(cfg, ros_cfg, perceptor, ros)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass