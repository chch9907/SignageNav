import time
import cv2
import numpy as np
from numpy.linalg import norm, inv, pinv
import os, sys
from PIL import Image
import yaml
import rospy
import pyrealsense2 as rs
from functools import partial
from easydict import EasyDict as edict
from collections import OrderedDict
from copy import copy
from geometry_msgs.msg import PointStamped
from datetime import datetime
import torch
import multiprocessing as mp
import networkx as nx
import pickle
from utils.utils import (
    read_yaml,
    intrinsics_to_dict,
    dict_to_intristics,
)
from scene_understand.ests import ESTS
from scene_understand.perceptor import Perceptor
from scene_understand.viewpoint import Viewpoint
from scene_understand.ocr_detector import OCR
from graph.topo_graph import Graph
from semantic_mapping.mapper import Mapper
from arguments import get_args
from utils.metrics import Metrics
from utils.ros import ROS
from utils.realsense import RealSenseCamera, RealSenseCameraDepth
import pyrealsense2 as rs 


LANDMARK_LABEL = 1
VIEWPOINT_LABEL = 2

def has_cht(rec):
    for c in rec:
        if '\u4e00' <= c <= '\u9fff':
            return True
    return False

def _init_output_dir(exp_id, cfg, ros_cfg):
    output_path = os.path.join(cfg.output_dir, exp_id)
    os.makedirs(output_path, exist_ok=True)
    with open(output_path + '/config.txt', 'a+') as f:
        f.write('cfg:' + str(cfg))
        f.write('\n')
        f.write('ros_cfg:' + str(ros_cfg))
    f.close()


depth_intrinsics = None


class Agent:
    def __init__(self, intrinsics_dict, cfg, ros_cfg, exp_id, rgbd_topics, debug):
        # print("----------start initialization----------")
        self.exp_id = exp_id
        self.cur_pose = 0
        self.ogmap = []
        self.debug = debug
        self.cfg = cfg
        self.prev_landmark = ""
        self.prev_localized_idx = 0
        self.min_ransac_num = cfg.min_ransac_num
        self.min_instance_count = cfg.min_instance_count
        self.area_3d_bound = cfg.area_3d_bound
        self.loc_type = cfg.loc_type
        self.loc_score_thred = cfg.loc_score_thred
        self.str_score_thred = cfg.str_score_thred
        self.output_path = os.path.join(cfg.output_dir, exp_id)
        self.encode = cfg.ocr_type == 'ESTS'
        self.plot = cfg.plot
        self.record = cfg.record
        self.render_freq = cfg.render_freq
        self.goal_frame = ros_cfg.frames.goal
        self.max_depth = ros_cfg.max_depth
        self.en_name_list = cfg.en_name_list
        self.final_goal = cfg.final_goal
        self.merge_dist_thred = cfg.merge_dist_thred
        self.ransacReprojThred = cfg.ransacReprojThred
        
        if cfg.record:
            self.rgb_output_path = os.path.join(cfg.output_dir, exp_id, 'rgbs')
            self.depth_output_path = os.path.join(cfg.output_dir, exp_id, 'depths')
            os.makedirs(self.rgb_output_path, exist_ok=True)
            os.makedirs(self.depth_output_path, exist_ok=True)
            self.metrics = Metrics(cfg, exp_id, copy(self.ros.mapData))
        self.subgoal_topic = ros_cfg.subgoal_topic
        self.next_subgoal_pub = rospy.Publisher(
            self.subgoal_topic, PointStamped, queue_size=10
        )
        
        ### Perceptor ###
        print('-' * 20)
        print("init perceptor")
        self.perceptor = Perceptor(cfg)

        ### ROS ###
        print('-' * 20)
        print("init ros_interface in agent process")
        
        depth_intrinsics = dict_to_intristics(intrinsics_dict)
        self.ros = ROS(cfg, ros_cfg, debug, depth_intrinsics, rgbd_topics=rgbd_topics, isMapping=True)

        ### viewpoint ###
        self.viewpoint_manager = Viewpoint(
            cfg, ros_cfg, self.perceptor, self.ros
        )  # include Perceptor
        self.viewpoint_range = ros_cfg.viewpoint_range
        self.viewpoint_score_thred = ros_cfg.viewpoint_score_thred

        ### Scene graph ###
        print('-' * 20)
        print("init inaccurate map")
        map_fig = cv2.imread(cfg.map_path)
        if os.path.exists(cfg.map_path.replace("png", "pkl")):
            self.venue_map = Graph(map_fig, cfg, is_global_map=True)
        else:
            assert False, 'please offline process venue map first.'
        self.map_landmarks = [item[0] for item in self.venue_map.node_landmarks]
        self.map_centers = np.array(self.venue_map.node_centers)
        self.map_centers[:, 1] = map_fig.shape[0] - self.map_centers[:, 1]
        print('map_landmarks:', self.map_landmarks)
        self.perceptor.scene_text_retrieval._map_landmarks = self.map_landmarks
        self.visited_dict = {lm: False for lm in self.map_landmarks}
        self.registered_landmarks = OrderedDict()
        self.registered_texts = []
        self.registered_poses = []
        self.localized_global_idxs = []
        self.localized_scores = []
        self.next_subgoal_world = []
        self.next_subgoal_idx = 0
        ## metric map
        print('-' * 20)
        print("init semantic mapper")
        self.semantic_map = Mapper(self.ros, cfg)  # tf listener
        self.contours = []
        
        print('-' * 20)
        print("successfully initialize, start running")
        rospy.sleep(3)
        
            
    def get_all_publishers(self,):
        return [
            self.next_subgoal_pub,
            self.viewpoint_manager.viewpoint_pub,
        ]

    def _wrap_msg(self, point_world):
        point_msg = PointStamped()
        point_msg.header.frame_id = self.goal_frame
        point_msg.header.stamp = rospy.Time().now()
        point_msg.point.x = point_world[0]
        point_msg.point.y = point_world[1]
        point_msg.point.z = point_world[2]
        return point_msg

    def pub_next_subgoal(self, next_subgoal_world):
        # for next_subgoal in next_subgoals_world:
        msg = self._wrap_msg(next_subgoal_world)
        self.next_subgoal_pub.publish(msg)
            
    
    def map_alignment(self, registered_poses, localized_global_idxs, localized_scores):
        """matching graph by considering geometric features"""
        ## address the cases where there are multiple the same localized texts.
        localized_scores = np.array(localized_scores)
        remain_list = []
        seen = []
        for global_idx in localized_global_idxs:
            if global_idx not in seen:
                seen.append(global_idx)
                hit_idxs = list(np.where(np.array(localized_global_idxs) == global_idx)[0])
                argmax_score_idx = np.argmax(localized_scores[hit_idxs])
                remain_list.append(hit_idxs[argmax_score_idx])
        ##! The input to cv2.estimateAffine2D must be in np.int64
        localized_global_idxs = np.array(localized_global_idxs)[remain_list].tolist()
        if len(localized_global_idxs) < self.min_ransac_num:
            return []
        print("localized_global_idxs:", localized_global_idxs)
        local_point_set = np.array(registered_poses, 
                                   dtype=np.int64)[remain_list, :2]
        global_point_set = self.map_centers[
            localized_global_idxs
        ].astype(np.int64)
        ## RANSAC + ICP
        retval, inliers = cv2.estimateAffinePartial2D(local_point_set, global_point_set,
                                                    cv2.USAC_MAGSAC, confidence = 0.95,
                                                    ransacReprojThreshold=self.ransacReprojThred, 
                                                    refineIters=10)
        transform = np.vstack((retval, np.array([0, 0, 1])))  # update transform matrix
        print('inliers:', inliers.T[0], "transform:", transform)

        
        all_node_centers = np.hstack((self.map_centers, 
                                np.array([[1] * len(self.map_centers)]).T)).T
        try:
            predicted_lmpose_world = inv(transform).dot(all_node_centers).T
            # print("predicted_lmpose_world:", predicted_lmpose_world.shape)  # (4, 2)
        except Exception as e:
            print(e)
            predicted_lmpose_world = pinv(transform).dot(all_node_centers).T
        
        predicted_lmpose_world = predicted_lmpose_world[:, :2] / predicted_lmpose_world[:, 2, None]

        ransac_data = {'raw_global': self.map_centers, 'local':local_point_set, 'global':global_point_set}
        with open('ransac_data.pkl', 'wb') as f:
            pickle.dump(ransac_data, f)
        return predicted_lmpose_world
        
        
    def run(self, queue, final_goal_idx=-1):
        # final_goal = self.final_goal
        # final_goal_idx = self.map_landmarks.index(final_goal)
        # final_goal_landmark = self.map_landmarks[final_goal_idx]
        # if final_goal_idx == -1:
        #     final_goal_idx = len(self.map_landmarks) - 1  # for searching route
        # next_subgoal_idx = final_goal_idx
        next_sub_landmark = ""
        finished = False
        reach_subgoal = False
        predicted_lmpose_world = []
        step = 0
        all_viewpoints = []
        
        tsp = nx.approximation.traveling_salesman_problem
        tsp_path = tsp(self.venue_map.graph, cycle=False)  # method: default to use christofides() for undirected graph
        visited_list = [False for _ in range(len(tsp_path))]
        print('tsp path:', tsp_path)
        while not rospy.is_shutdown():
            if self.plot and self.ros.new_nbv:
                print('render next_best_pose')
                cur_pose = self.ros.nbv_cur_pose
                contours = self.semantic_map.get_clustered_contours(
                    cur_pose, step, show_contours=False, show_clusters=False
                )
                self.semantic_map.project_semantics(
                    self.registered_poses, self.registered_texts, contours, all_viewpoints, 
                    predicted_lmpose_world, self.next_subgoal_world, self.next_subgoal_idx, cur_pose, step, self.exp_id, 
                    plot=self.plot, is_nbv=self.ros.new_nbv
                )  # self.registered_landmarks
                self.ros.new_nbv = False
            if queue.empty():
                continue
            (cur_pose, ocr_result, depth_img, depth_colorimg) = queue.get()
            ######### 1) Perception  #########
            # print("--------------step:", step, "--------------")
            
            if self.cfg.record:
                rospy.loginfo(f'save image at step {step}')
                # cv2.imwrite(self.rgb_output_path + f'/{step}.png', color_img)
                cv2.imwrite(self.depth_output_path + f'/{step}.png', depth_img)
                
            print('-' * 20)
            print("agent step:", step)
            text_list = [item['text'] for item in ocr_result]
            centers_map, centers_world, distances, areas_3d, filter_idxs = self.ros.get_region_depth(
                depth_img, ocr_result, cur_pose, 
                filter_by_area_3d=True, area_3d_bound=self.area_3d_bound
            )

            new_instances, new_idxs, fuse_idxs = self.perceptor.semantic_fusion(
                centers_world, centers_map, ocr_result
            )
            new_centers_world = np.array(centers_world)[new_idxs]
            new_centers_map = np.array(centers_map)[new_idxs]
            new_distances = np.array(distances)[new_idxs]
            fuse_poses = self.perceptor.get_fuse_poses(fuse_idxs)
            # fuse_poses_map = self.perceptor.get_fuse_poses_map(fuse_idxs)
            
 
            ######### 2) 3D scene text retrieval  #########
            if not len(self.localized_global_idxs):  # if no localized landmark yet
                neighbor_lm_list = self.map_landmarks
            else:
                neighbor_idxs = self.venue_map.node_neighbors[localized_global_idx]
                neighbor_lm_list = [self.map_landmarks[idx] for idx in neighbor_idxs]
            loc_idxs, loc_texts, match_scores = self.perceptor.retrieval(fuse_idxs,
                                                                         neighbor_lm_list,
                                                                         _type=self.loc_type)


            ######### 3) generate viewpoint  #########
            viewpoint_valid_list = [idx for idx in range(len(new_distances)) 
                                    if new_distances[idx] <= self.viewpoint_range and idx in loc_idxs
                                    and match_scores[loc_idxs.index(idx)] >= self.viewpoint_score_thred]
            
            if len(viewpoint_valid_list):
                _, viewpoints_world, _ = self.viewpoint_manager.update(
                    [new_instances[idx] for idx in viewpoint_valid_list],
                    new_centers_world[viewpoint_valid_list],
                    new_centers_map[viewpoint_valid_list],
                    new_distances[viewpoint_valid_list],
                    depth_img,
                    depth_colorimg,
                    cur_pose,
                    vis=False,
                )
                viewpoints_score = [match_scores[loc_idxs.index(idx)] for idx in viewpoint_valid_list]
                self.viewpoint_manager.pub_viewpoints(viewpoints_world, viewpoints_score)
                print('pub viewpoint:', viewpoints_world, 'dist:', new_distances)
                all_viewpoints.extend(viewpoints_world)
            else:
                if len(new_distances):
                    print('viewpoint exceeds range or score:', new_distances)
                viewpoints_world = []

            
            ######### 4) Landmark map update #########
            topk_text_list = []
            real_loc_texts = []
            new_real_loc_texts = []
            for loc_idx, topk_text, topk_score, fuse_pose in zip(
                loc_idxs, loc_texts, match_scores, fuse_poses
            ):
                loc_text = topk_text
                match_score = topk_score.detach().cpu()
                topk_text_list.append(copy(topk_text))
                try:
                    if self.perceptor.instance_counts[fuse_idxs[loc_idx]] >= self.min_instance_count:
                        pass
                except:
                    print('list index out of range:', loc_idx, fuse_idxs, len(self.perceptor.instance_counts))
                if self.perceptor.instance_counts[fuse_idxs[loc_idx]] >= self.min_instance_count \
                    and match_score >= self.loc_score_thred:
                   
                    print(f"localize on {loc_text} with similarity {match_score:.3f}! hit count:{self.perceptor.instance_counts[fuse_idxs[loc_idx]]}")
                    
                    localized_global_idx = self.map_landmarks.index(loc_text)

                    isNew = True
                    real_loc_texts.append(loc_text)
                    self.prev_landmark = loc_text
                    
                    if len(self.registered_poses):
                        registered_dists = np.linalg.norm(fuse_pose[:2] - np.array(self.registered_poses)[:, :2],
                                    axis=1
                        )
                        if np.min(registered_dists) <= self.merge_dist_thred:  #! not consider similarity
                            merge_idx = np.argmin(registered_dists)
                            self.registered_poses[merge_idx] = fuse_pose
                            self.registered_texts[merge_idx] = loc_text
                            isNew = False
                            #! correct previous localization
                            visited_list[self.localized_global_idxs[merge_idx]] = False  
                            self.localized_global_idxs[merge_idx] = localized_global_idx
                            self.localized_scores[merge_idx] = match_score
                            ## only change fuse_pose, reserve loc_text label
                    if isNew:
                        self.registered_poses.append(fuse_pose)
                        self.registered_texts.append(loc_text)
                        new_real_loc_texts.append(loc_text)
                        self.localized_global_idxs.append(localized_global_idx)
                        self.localized_scores.append(match_score)
                    visited_list[localized_global_idx] = True

            
            ######### 5) Route search on venue map #########
            next_subgoal_idx = None
            if len(real_loc_texts):
                ## tsp 
                start = tsp_path.index(localized_global_idx)
                remained_tsp_path = [100 for _ in range(len(tsp_path))]
                for i in range(len(tsp_path)):
                    if visited_list[tsp_path[i]] == False:
                        remained_tsp_path[i] = abs(i - start)
                print('remained_tsp_path:', remained_tsp_path)
                next_subgoal_idx = tsp_path[np.argmin(remained_tsp_path)]
                if np.min(remained_tsp_path) == 100:
                    print('finish mapping all the landmarks')
                    break

                if next_subgoal_idx is not None:
                    next_sub_landmark = self.map_landmarks[next_subgoal_idx]
                    print("next subgoal landmark:", next_sub_landmark)
                
                    ## project goal from guide map to online-built ogmap
                    if  len(self.localized_global_idxs) >= self.min_ransac_num:
                        print('map alignment:')
                        predicted_lmpose_world = self.map_alignment(self.registered_poses, 
                                                                    self.localized_global_idxs,
                                                                    self.localized_scores)  # transform, inliers
                        if len(predicted_lmpose_world):
                            next_subgoal_world = predicted_lmpose_world[next_subgoal_idx].tolist()
                            next_subgoal_world += [next_subgoal_idx]  # [x, y, idx]
                            print('next subgoal world:', next_subgoal_world)
                            if next_subgoal_world != self.next_subgoal_world:
                                self.pub_next_subgoal(next_subgoal_world)
                                self.next_subgoal_world = next_subgoal_world
                                self.next_subgoal_idx = next_subgoal_idx
            

            ######### 6) Update semantic occupancy map  #########
            if self.plot and len(real_loc_texts): # len(new_real_loc_texts)
                print('render semantic map')
                contours = self.semantic_map.get_clustered_contours(
                    cur_pose, step, show_contours=False, show_clusters=False
                )
                self.semantic_map.project_semantics(
                    self.registered_poses, self.registered_texts, contours, all_viewpoints, 
                    predicted_lmpose_world, self.next_subgoal_world, self.next_subgoal_idx, cur_pose, step, self.exp_id, 
                    plot=self.plot, is_nbv=False
                )  # self.registered_landmarks
                
            
            step += 1
            self.ros.rate.sleep()
        
        print("finish navigation.")


def detector_process(ESTS_model, camera_obj, rgbd_topics, 
                     cfg, ros_cfg, exp_id, queue_ocr, 
                     show=False, save_res=False):
    ## pyrealsense object cannot be pickled and sent to subprocess, thus used in main process. 
    print('starting detector process')
    rospy.init_node('my_detector')
    scene_text_detector = OCR(cfg, ESTS_model)
    print("init ros_interface in detector process")
    ros = ROS(cfg, ros_cfg, ros_cfg.debug, isMapping=False, camera_obj=camera_obj, rgbd_topics=rgbd_topics)
    encode = cfg.ocr_type == 'ESTS'
    step = -1
    st = time.time()
    count = 0
    rospy.loginfo('wait for rgb and depth')
    while not rospy.is_shutdown():
        color_img, depth_img, depth_colormap = (
            ros.get_image()
        )
        if color_img is not None and depth_img is not None:
            break
        
    try:
        while not rospy.is_shutdown():
            color_img, depth_img, depth_colormap = (
                ros.get_image()
            )  # obs: (color, depth, color_depth)
            if color_img is None:
                continue
            step += 1
            cur_pose = ros.get_tf()
            st = time.time()
            ocr_result = scene_text_detector(  # 1.6s per frame
               color_img, save_res=save_res, encode=encode
            )
            rospy.loginfo(f'detector step:{step}, ocr time:{time.time() - st}')
            count += 1
            valid_res = ocr_result
            if len(valid_res):
                print('detect res:', [item[1] for item in valid_res])
                # print(cur_pose, ocr_result, type(depth_img), type(depth_colormap))
                queue_ocr.put((cur_pose, valid_res, depth_img, depth_colormap))
                
            if show:
                for res in ocr_result:
                    xl, yl, xr, yr = res['bbox'].astype(np.int64)
                    cv2.rectangle(color_img, (xl, yl), (xr, yr), (255, 0, 0), 2)  # red color
                cv2.imshow(ros_cfg.camera_id, color_img)  # depth_colormap
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    pass
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
    ros.close_camera()
    print("destroy detector process")


def batch_detector_process(ESTS_model, queue_imgs, queue_ocr, cfg, ros_cfg, exp_id, 
                      show=False, save_res=False):
    ## pyrealsense object cannot be pickled and sent to subprocess, thus used in main process. 
    print('init batch detector process...')
    rospy.init_node('my_detector')
    scene_text_detector = OCR(cfg, ESTS_model)
    encode = cfg.ocr_type == 'ESTS'
    step = 0
    if cfg.record:
        rgb_output_path = os.path.join(cfg.output_dir, exp_id, 'rgbs')
        depth_output_path = os.path.join(cfg.output_dir, exp_id, 'depths')
        os.makedirs(rgb_output_path, exist_ok=True)
        os.makedirs(depth_output_path, exist_ok=True)
    st = time.time()
    k = 1
    try:
        while not rospy.is_shutdown():
            if queue_imgs.empty():
                continue
            batch_imgs = queue_imgs.get()
            batch_color_img = [item[0] for item in batch_imgs]
            batch_ocr_result = scene_text_detector(  # 1.6s per frame
               batch_color_img, save_res=save_res, encode=encode
            )
            sorted_ocr_result = [[] for _ in range(len(batch_imgs))]
            for res in batch_ocr_result:
                sorted_ocr_result[res['batch_idx']].append(res)
            batch_regions = []
            for ocr_result, (color_img, depth_img, depth_colormap, cur_pose) in zip(sorted_ocr_result, batch_imgs):
                valid_res = ocr_result
                if show:
                    for res in valid_res:
                        xl, yl, xr, yr = res['bbox'].astype(np.int64)
                        cv2.rectangle(color_img, (xl, yl), (xr, yr), (255, 0, 0), 2)  # red color
                    cv2.imshow(ros_cfg.camera_id, color_img)  # depth_colormap
                    if cv2.waitKey(3) & 0xFF == ord("q"):
                        pass
                if len(valid_res):
                    queue_ocr.put((cur_pose, valid_res, depth_img, depth_colormap))
                        
                
                if cfg.record:
                    print('save image')
                    cv2.imwrite(rgb_output_path + f'/{step}.png', color_img)
                    cv2.imwrite(depth_output_path + f'/{step}.png', depth_img)
            step += 1
    except rospy.ROSInterruptException:
        pass
    # cv2.destroyAllWindows()
    # ros.close_camera()
    print("destroy detector process")
    

def camera_process(queue_imgs, camera_obj, rgbd_topics, cfg, ros_cfg, max_buffer_size=1, rateHz=1):
    rospy.init_node('my_camera')
    print('init camera process...')
    ros = ROS(cfg, ros_cfg, ros_cfg.debug, isMapping=False, camera_obj=camera_obj, rgbd_topics=rgbd_topics)
    buffer = []
    step = 0
    rate = rospy.Rate(rateHz)
    
    rospy.loginfo('wait for rgb and depth')
    while not rospy.is_shutdown():
        color_img, depth_img, depth_colormap = (
            ros.get_image()
        )
        if color_img is not None and depth_img is not None:
            break
    pre_color_img = color_img
    st = time.time()
    try:
        while not rospy.is_shutdown():
            color_img, depth_img, depth_colormap = (
                ros.get_image()
            )  # obs: (color, depth, color_depth)
            
            # if ros_cfg.show:
            #     cv2.imshow(ros_cfg.camera_id, color_img)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         pass
            if color_img is None or (color_img == pre_color_img).all(): 
                continue
            pre_color_img = color_img
            cur_pose = ros.get_tf()
            buffer.append((color_img, depth_img, depth_colormap, copy(cur_pose))) 
            if len(buffer) == max_buffer_size:
                queue_imgs.put(buffer)
                buffer = []
                print(f'camera step:{step}, buffer time:{time.time() - st}') 
                st = time.time() 
                step += 1
            rate.sleep()
            
            
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
    ros.close_camera()
    print("destroy camera process")


def agent_process(queue_ocr, intrinsics_dict, cfg, ros_cfg, rgbd_topics, exp_id, debug):
    print('starting agent process')
    rospy.init_node('my_agent')
    try:
        agent = Agent(intrinsics_dict, cfg, ros_cfg, exp_id, rgbd_topics, debug)
        agent.run(queue_ocr)
    except rospy.ROSInterruptException:
        pass
    print('destroy agent process')


if __name__ == '__main__':
    args = get_args()
    if args.scene == '1':
        config_path = './config/scene1.yaml'
    elif args.scene == '2':
        config_path = './config/scene2.yaml'
    else:
        raise ValueError(args.scene)
    print('scene:', args.scene)
    cfg = edict(read_yaml(config_path))
    cfg.update(vars(args))
    ros_cfg = edict(read_yaml(args.ros_config_path))
    ros_cfg.update(vars(args))
    print('cfg:', cfg)
    print('ros_cfg:', ros_cfg)
    print('-' * 20)
    camera_obj = None
    # depth_intrinsics = None
    # intrinsics_dict = None
    rgbd_topics = None
    if args.use_camera_api:
        print('camera mode: use_camera_api')
        camera_obj = RealSenseCameraDepth(name=ros_cfg.camera_id, fps=ros_cfg.camera_fps)
        depth_intrinsics = camera_obj.depth_intrinsics
        intrinsics_dict = intrinsics_to_dict(depth_intrinsics)
    elif args.use_camera_topic:
        print('camera mode: use_camera_topic')
        rgbd_topics = {
            'rgb': '/camera/color/image_raw',
            'depth': '/camera/aligned_depth_to_color/image_raw',
            # 'camera_info': 'camera/aligned_depth_to_color/camera_info'
        }
        intrinsics_dict = ros_cfg.camerae_intrinstics
    else:
        raise ValueError('please specify mode: {use_camera_api or use_camera_topic}')
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    print("init ESTS")
    ESTS_model = ESTS(cfg)
    
    
    exp_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
    _init_output_dir(exp_id, cfg, ros_cfg)
    
    #============= single-inference mode =============
    # queue_ocr = mp.Queue()
    # p_agent = mp.Process(target=agent_process, args=(queue_ocr, intrinsics_dict, cfg, ros_cfg, rgbd_topics, exp_id, args.debug), daemon=True)
    # p_agent.start()
    
    # detector_process(ESTS_model, camera_obj, rgbd_topics,
    #                  cfg, ros_cfg, exp_id, queue_ocr, args.show) 
    # p_agent.terminate()
    # p_agent.join()
    
    
    #============= batch-inference mode =============
    queue_imgs = mp.Queue(cfg.max_buffer_size + 1)
    queue_ocr = mp.Queue()
    
    p_camera = mp.Process(target=camera_process, 
                        args=(queue_imgs, camera_obj, rgbd_topics, cfg, ros_cfg, 
                              cfg.max_buffer_size, cfg.rateHz), daemon=True)
    p_agent = mp.Process(target=agent_process, 
                         args=(queue_ocr, intrinsics_dict, cfg, 
                               ros_cfg, rgbd_topics, exp_id, args.debug), 
                         daemon=True)
    
    p_camera.start()
    p_agent.start()

    batch_detector_process(ESTS_model, queue_imgs, queue_ocr,
                     cfg, ros_cfg, exp_id, args.show) 

    p_camera.terminate()
    p_camera.join()
    
    p_agent.terminate()
    p_agent.join()
    
    
    
    ## clean output files for failed experiments 
    output_path = os.path.join(cfg.output_dir, exp_id)
    if os.path.exists(output_path) and len(os.listdir(output_path)) <= 2:
        os.system(f'rm -r {output_path}')
    
