import numpy as np
from numpy.linalg import norm
import re
import random
from math import pi
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher
from Levenshtein import ratio
import yaml
import skimage
from typing import Dict
from enum import Enum
import pickle
import cv2
import copy
from queue import Queue
from sklearn.neighbors import NearestNeighbors
import pyrealsense2 as rs 
import rospy  # type:ignore

OBSTACLE = 100
UNKNOWN = -1
FREE = 0

chn_list = []
with open('./scene_understand/ESTextSpotter/chn_cls_list.txt', 'rb') as fp:
    CTLABELS = pickle.load(fp)
    chn_list = [str(chr(c)) for c in CTLABELS]
    
def _str_filter(rec):
    s = ''
    for c in rec:
        if c in chn_list:
            s += c
    return s


remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~0-9]'
filter_list = ['一', '出口']
def OCR_filter(result, prob_thred, min_area=1000, min_char=1, min_chn=1, max_chn=3):
    bbox, text, prob = result
    bbox = np.array(bbox)
    text = re.sub(remove_chars, "", text)  # remove special chars
    remove_str = ''
    for c in text:  # remove circled_number
        if c.isdigit() and not c.isdecimal():
            remove_str += c
    text = re.sub(remove_str, "", text)
    if text in filter_list:
        print('filter:', text)
        text = ''
    if len(bbox.shape) == 1:
        xl, yl, xr, yr = bbox
    else:
        xl, yl, xr, yr = bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1]
    area = (xr - xl) * (yr - yl)
    
    enough_text = (not has_chn(text) and len(text) >= min_char) or \
                (has_chn(text) and min_chn <= len(text) <= max_chn)
    if enough_text and prob >= prob_thred and area >= min_area: 
        result[1] = text
        return result
    else:
        return None

def has_chn(rec):
    for c in rec:
        if '\u4e00' <= c <= '\u9fff': # chn
            return True
    return False
    
def judge_overlap(l1, r1, l2, r2):
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        return False
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        return False
    # If one rectangle is above other
    if r1.y > l2.y or r2.y > l1.y:
        return False
    return True

def get_center_dist(ct1, ct2):
    '''can be used to calculate 2D or 3D distance'''
    if not isinstance(ct1, np.ndarray):
        ct1 = np.array(ct1)
    if not isinstance(ct2, np.ndarray):
        ct2 = np.array(ct2)  
    return norm(ct1 - ct2)

def get_horizontal_dist(xl1, xr1, xl2, xr2):
    assert xr1 > xl1 and xr2 > xl1
    if xr1 <= xl2:  # boxA is on the right of boxB
        h_dist = xl2 - xr1
    elif xl1 >= xr2: # boxA is on the right of boxB
        h_dist = xl1 - xr2
    else:
        h_dist = -1  # intersection or incorporation
    return h_dist

def bb_intersection_over_union(boxA, boxB):
    '''input: (xl, yl, xr, yr)'''
    if isinstance(boxA, np.ndarray) and len(boxA.shape) == 2:
        boxA = [boxA[0, 0], boxA[0, 1], boxA[2, 0], boxA[2, 1]]
    if isinstance(boxB, np.ndarray) and len(boxB.shape) == 2:
        boxB = [boxB[0, 0], boxB[0, 1], boxB[2, 0], boxB[2, 1]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except:
        print('divide by zero:', boxAArea, boxBArea, interArea)
        assert False
    return iou # return the intersection over union value
    

def union(current_box, boxB):  # (lefttop, rightbuttion)
    current_box[0] = min(current_box[0], boxB[0])
    current_box[1] = min(current_box[1], boxB[1])
    current_box[2] = max(current_box[2], boxB[2])
    current_box[3] = max(current_box[3], boxB[3])
    return current_box

def get_center(bbox):
    xl, yl, xr, yr = bbox
    return [(xr - xl) / 2, (yr - yl) / 2]

def join_priority(centerA, centerB, height_ths=1):
    '''judge priority of the joined text in image coordinate
    topleft >= buttomright
    '''
    yA, xA = centerA
    yB, xB = centerB 
    if abs(yA - yB) >= height_ths:   # first consider vertical distance
        if yA <= yB:
            return True  
        else:
            return False
    else:   # if at the near level, then consider horizon distance
        if xA <= xB:
            return True
        else:
            return False

def _str_similar(a, b):
    isjunk = None  # lambda x: x in " \\t"  # blanks and indent
    return SequenceMatcher(isjunk, a, b).ratio()


def merge_string(stringA, stringB, priority, sim_thred=0.6):
    str_sim = _str_similar(stringA, stringB)
    if str_sim >= sim_thred:
        return stringA if len(stringA) >= len(stringB) else stringB
    else:
        return stringA + stringB if priority else \
            stringB + stringA

def get_elements_by_idxs(elements, idxs):
    if isinstance(elements, np.ndarray):
        return elements[idxs]
    elif isinstance(elements, list):
        return [elements[i] for i in idxs]

def non_maximum_suppress(pred_results, features, batch_idxs, iou_thred=None):
    pred_results = [pred + [feat, idx] for (pred, feat, idx) in zip(pred_results, features, batch_idxs)]
    sorted_pred_results = sorted(pred_results, key=lambda x: x[2], reverse=True)  # descend by probs
    keep = np.ones(len(sorted_pred_results), dtype=bool)
    nms_res = []
    nms_features = []
    nms_batch_idxs = []
    for i, (bbox, label, prob, feat, idx) in enumerate(sorted_pred_results):
        if not keep[i]:
            continue
        nms_res.append([bbox, label, prob])
        nms_features.append(feat)
        nms_batch_idxs.append(idx)
        bbox_i = sorted_pred_results[i][0]
        label_i = sorted_pred_results[i][1]
        for j in range(i + 1, len(sorted_pred_results)):
            if iou_thred is not None:
                bbox_j = sorted_pred_results[j][0]
                keep[j] = ~(bb_intersection_over_union(bbox_i, bbox_j) >= iou_thred)
            else:
                label_j = sorted_pred_results[j][1]
                keep[j] = ~(label_j == label_i)
    return nms_res, nms_features, nms_batch_idxs


def factorize(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    return factor


def random_crop(image, scale=0.4):
    # crop image
    height, width = int(image.shape[0]*scale), int(image.shape[1]*scale)
    x = random.randint(0, image.shape[1] - int(width))
    y = random.randint(0, image.shape[0] - int(height))
    cropped = image[y:y+height, x:x+width]
    return cropped, (y, x)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def _normalize_heading(heading):
    if heading > pi:
        heading -= 2 * pi
    elif heading < -pi:
        heading += 2 * pi
    return heading

def _calc_distance(dx, dy):
    return np.square(dx ** 2 + dy ** 2)

def get_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def read_map_pgm(prefix, vis=False)-> np.ndarray:
    im = Image.open(prefix + '.pgm')
    if vis:
        im.show()
    return np.array(im)
    
def read_map_yaml(prefix)-> Dict:
    '''
    image: testmap.png
    resolution: 0.1  # meters/pixel
    origin: [0.0, 0.0, 0.0]  # the 2D pose of the left-bottom pixel:(x,y,yaw),yaw rotate in anti-clockwise(yaw is often ignored)。
    occupied_thresh: 0.65  # exceed
    free_thresh: 0.196  # below
    negate: 0: whether reverse the semantic labels
    '''
    config = yaml.load(open(prefix + '.yaml'), Loader=yaml.FullLoader)
    return config

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def mid_perpendicular_lines(x1, y1, x2, y2):
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = x1**2 - x2**2 + y1**2 - y2**2
    return A, B, C

def get_viewpoints(boxes, depth_map, afford_dist=2):
    xl, yl, xr, yr = boxes
    depth1 = depth_map[int((yr - yl) / 2), int(xl)]
    depth2 = depth_map[int((yr - yl) / 2), int(xr)]
    A, B, C = mid_perpendicular_lines(xl, depth1, xr, depth2)
    theta = np.arctan(-A / B)
    viewpoint = afford_dist * np.array([np.cos(theta), np.sin(theta)])  # x, z
    return viewpoint
    
def XtoY(setX, transform):
    assert isinstance(setX, np.ndarray)
    _setX = np.hstack((setX, np.array([[1] * setX.shape[0]]).T)).T  
    new_setX = transform.dot(_setX)  # homogeneous transform
    return new_setX

def get_new_pose(pose, rel_pose_change):
    """taken from chaplot's ANS: 
    pose is in world coordinate.
    rel_pose_change is in local coordinate.
    """
    if len(pose.shape) > 1:
        x, y, o = pose[:, 0], pose[:, 1], pose[:, 2]
        dx, dy, do = rel_pose_change[:, 0], rel_pose_change[:, 1], rel_pose_change[:, 2]
    else:
        x, y, o = pose
        dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(o) + dy * np.cos(o)
    global_dy = dx * np.cos(o) - dy * np.sin(o)
    x += global_dy
    y += global_dx
    o += do

    if len(pose.shape) > 1:
        for i in range(len(o)):
            o[i] = _normalize_heading(o[i])
        return np.stack([x, y, o], axis=1)
    else:
        o = _normalize_heading(o)
        return np.array([x, y, o])

def pixel_to_world(depth_intrinsics, center_2d, distance):
    '''coordinate definition:
    https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0?fbclid=IwAR3gogVZe824YUps88Dzp02AN_XzEm1BDb0UbmzfoYvn1qDFb7KzbIz9twU#pixel-coordinates
    input: pixel coordinate
    output: point coordinate
    '''
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, center_2d, distance)  # x, y, depth
    # point[0], point[1], point[2] = right, down, forward
    # transform to SLAM right-hand coordinate: x, y, z = point[2], -point[0], -point[1]
    return [point[2], -point[0], -point[1]]

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def neighbor_bfs_free(map_array, point_map, dir_num = 8):
    free = 0  # free
    ay, ax = mx, my = point_map  # array coord
    ori_point = (ax, ay)
    if dir_num == 4:
        search_dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 dir
    elif dir_num == 8:
        search_dir = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    else:
        raise ValueError(dir_num)
    free_y, free_x = -1, -1
    h, w = map_array.shape
    que_bfs = Queue()
    que_bfs.put(ori_point)
    visit_list = []
    find_target = False
    free_points = []
    while not que_bfs.empty():
        p = que_bfs.get()
        if p in visit_list:
            continue
        if 0 <= p[0] < h and 0 <= p[1] < w and map_array[p] == free:
            # free_y, free_x = p
            # return (free_x, free_y)  # map coord
            free_points.append(list(p))
        visit_list.append(p)
        for (dx, dy) in search_dir:
            new_p = (p[0] + dx, p[1] + dy)
            if 0 <= new_p[0] < h and 0 <= new_p[1] < w and new_p not in visit_list:
                if map_array[new_p] != free:
                    que_bfs.put(new_p)
                else:
                    # free_y, free_x = new_p
                    # return (free_x, free_y)  # map coord
                    free_points.append(list(new_p))
        if len(free_points):
            break
    if len(free_points) == 1:
        return (free_points[0][1], free_points[0][0])  # free_x, free_y
    elif len(free_points) > 1:  #!bug: nearest point may still not reachable
        nearest_idx = np.argmin(norm(np.array(ori_point) - 
                                     np.array(free_points), axis=1))
        nearest_point = free_points[nearest_idx]
        return (nearest_point[1], nearest_point[0])  # y, x
    else:
        print('error: could not find free viewpoint:', point_map)
        return None


def _world_to_map(_map, world_pose):
    wx, wy = world_pose[:2]
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        print("World coordinates out of bounds")  # raise Exception
        return None

    mx = int((wx - _map.info.origin.position.x) / _map.info.resolution)
    my = int((wy - _map.info.origin.position.y) / _map.info.resolution)
    
    if  (my > _map.info.height or mx > _map.info.width):
        print("Map height or width out of bounds:", my, _map.info.height, mx, _map.info.width)  #raise Exception
        return None
    return (mx, my) # to np.array index  

def _world_to_map2(_map, world_pose, predict=False):
    wx, wy = world_pose[:2]
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        print("World coordinates out of bounds")  # raise Exception
        if not predict:
            return None
    mx = int((wx - _map.info.origin.position.x) / _map.info.resolution)
    my = int((wy - _map.info.origin.position.y) / _map.info.resolution)
    if  (my > _map.info.height or mx > _map.info.width):
        print("Map height or width out of bounds:", my, _map.info.height, mx, _map.info.width)  #raise Exception
        if not predict:
            return None
    return (my, mx) # to np.array index  

def _map_to_world(_map, map_pose):
    my, mx = map_pose 
    wy = (my + 0.5) * _map.info.resolution + _map.info.origin.position.y
    wx = (mx + 0.5)  * _map.info.resolution + _map.info.origin.position.x
    world_pose = [wx, wy]
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        rospy.logwarn(f"{world_pose}:World coordinates out of bounds")
    if  my > _map.info.height or mx > _map.info.width:
        rospy.logwarn(f"{world_pose}:Map height or width out of bounds")
    return world_pose


def intrinsics_to_dict(intrinstics_obj):
    return {
        'width': intrinstics_obj.width,
        'height': intrinstics_obj.height,
        'ppx': intrinstics_obj.ppx,
        'ppy': intrinstics_obj.ppy,
        'fx': intrinstics_obj.fx,
        'fy': intrinstics_obj.fy,
        'model': intrinstics_obj.model,
        'coeffs': intrinstics_obj.coeffs
    }

def dict_to_intristics(dict_):
    depth_intrinsic = rs.pyrealsense2.intrinsics()
    depth_intrinsic.width = dict_['width']
    depth_intrinsic.height = dict_['height']
    depth_intrinsic.ppx = dict_['ppx']
    depth_intrinsic.ppy = dict_['ppy']
    depth_intrinsic.fx = dict_['fx']
    depth_intrinsic.fy = dict_['fy']
    depth_intrinsic.model = dict_['model']  #rs.distortion.inverse_brown_conrady
    depth_intrinsic.coeffs = dict_['coeffs']
    return depth_intrinsic

def image_blur_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def has_cht(text):
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            return True
    return False

def _sim_one_to_multiple(local_lm, global_lm):
    return [_str_similar(local_lm, lm) for lm in global_lm]

def compute_normal(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return normal