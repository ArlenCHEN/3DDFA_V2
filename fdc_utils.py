import numpy as np
import yaml
import sys

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose, viz_pose_1
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

from PIL import Image

is_debug = False

def detect_attributes(img, args):
    """
    Detect facial attributes of the given images, including 2/3D landmarks, head pose, depth, etc
    See this for more details: https://github.com/cleardusk/3DDFA_V2
    """
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        return None
    else:
        param_lst, roi_box_lst = tddfa(img, boxes)

        # Visualization and serialization
        dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
        
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

        if is_debug:
            print('ver_lst: ', ver_lst)

        # Only return 2D landmarks
        return ver_lst[0]

def recrop_img(img, scale_factor):
    '''
    input img: PIL image
    '''    
    full_size, _ = img.size
    margin_size = int((full_size - scale_factor*full_size)*0.5)
    far_end_size = int(full_size - margin_size)
    recrop_size = (margin_size, margin_size, far_end_size, far_end_size)
    cropped_img = img.crop(recrop_size)
    new_size = (full_size, full_size)
    new_img = cropped_img.resize(new_size, resample=Image.BILINEAR)

    # Return a new numpy image rescaled
    return np.asarray(new_img)

def detect_face(img, landmark_raw):
    """
    Crop face area according to the detected landmarks
    """
    # # This size must ensure the face part is not missing, otherwise, the landmark detection will generate negative coordinates
    # # img_size = 1000 # Hyperparam 1000 for video_001
    # v_size = 700
    u_size = 1080
    # if is_debug:
    #     print('In detect_face, shape of input img: ', img.shape)

    u_coords = landmark_raw[0,:]
    # v_coords = landmark_raw[1,:]
    u_max = int(np.max(u_coords))
    u_min = int(np.min(u_coords))
    # v_max = int(np.max(v_coords))
    # v_min = int(np.min(v_coords))

    middle_u = int((u_max+u_min)/2)
    # middle_v = int((v_max+v_min)/2)

    # print('original image shape: ', img.shape)

    # new_v_min = middle_v - int(v_size/2)
    # new_v_max = new_v_min + v_size
    # assert new_v_max < img.shape[0], 'new_v_max exceeds the boundary'
    
    new_u_min = middle_u - int(u_size/2)
    new_u_max = new_u_min + u_size

    if new_u_min < 0:
        new_u_min = 0

    if new_u_max > img.shape[1]:
        new_u_max = img.shape[1]

    # print(new_u_min, new_u_max)
    # assert new_u_max < img.shape[1], 'new_u_max exceeds the boundary'

    # if new_u_max < 0 or new_u_min < 0 or new_v_max < 0 or new_v_min < 0:
    #     print('Index: ', new_u_max, new_u_min, new_v_max, new_v_min)
    #     raise RuntimeError('Index is negative!')

    # new_img = img[new_v_min:new_v_max, new_u_min:new_u_max, :]
    # print('In detect_face, output shape: ', new_img.shape)
    # # input()

    new_img = img[:,new_u_min:new_u_max,:] # Crop the image to [1080, 1080]
    # scale_factor = 0.7
    # new_img = Image.fromarray(new_img) # Convert numpy image to PIL image
    # new_img = recrop_img(new_img, scale_factor)

    return new_img
    
def split_face(landmark_raw):
    """
    Crop left eyes, right eyes, and mouth from the given image
    """

    left_eyebrow = np.hstack((landmark_raw[0, 17:22][:, np.newaxis], landmark_raw[1, 17:22][:, np.newaxis]))
    if is_debug:
        print('Left eyebrow: ', left_eyebrow)

    right_eyebrow = np.hstack((landmark_raw[0, 22:27][:, np.newaxis], landmark_raw[1, 22:27][:, np.newaxis]))
    if is_debug:
        print('Right eyebrow: ', right_eyebrow)

    left_eye = np.hstack((landmark_raw[0, 36:42][:, np.newaxis], landmark_raw[1, 36:42][:, np.newaxis]))
    if is_debug:
        print('Left eye: ', left_eye)

    right_eye = np.hstack((landmark_raw[0, 42:48][:, np.newaxis], landmark_raw[1, 42:48][:, np.newaxis]))
    if is_debug:
        print('Right eye: ', right_eye)

    mouth = np.hstack((landmark_raw[0, 48:68][:, np.newaxis], landmark_raw[1, 48:68][:, np.newaxis]))
    if is_debug:
        print('Mouth: ', mouth)

    left = np.vstack((left_eyebrow, left_eye))
    if is_debug:
        print('Left part: ', left)

    right = np.vstack((right_eyebrow, right_eye))
    if is_debug:
        print('Right part: ', right)

    # Right part if the image itself = Left part of the third-person view
    right_u_max = int(np.max(left[:, 0])) 
    right_u_min = int(np.min(left[:, 0]))
    right_v_max = int(np.max(left[:, 1]))
    right_v_min = int(np.min(left[:, 1]))

    # Left part if the image itself = Right part of the third-person view
    left_u_max = int(np.max(right[:, 0]))
    left_u_min = int(np.min(right[:, 0]))
    left_v_max = int(np.max(right[:, 1]))
    left_v_min = int(np.min(right[:, 1]))

    if is_debug:
        print('Left u max: ', left_u_max)
        print('Left u min: ', left_u_min)
        print('Left v max: ', left_v_max)
        print('Left v min: ', left_v_min)

    
    if is_debug:
        print('Right u max: ', right_u_max)
        print('Right u min: ', right_u_min)
        print('Right v max: ', right_v_max)
        print('Right v min: ', right_v_min)

    mouth_u_max = int(np.max(mouth[:, 0]))
    mouth_u_min = int(np.min(mouth[:, 0]))
    mouth_v_max = int(np.max(mouth[:, 1]))
    mouth_v_min = int(np.min(mouth[:, 1]))
    if is_debug:
        print('Mouth u max: ', mouth_u_max)
        print('Mouth u min: ', mouth_u_min)
        print('Mouth v max: ', mouth_v_max)
        print('Mouth v min: ', mouth_v_min)

    left_coords = [left_v_max, left_v_min, left_u_max, left_u_min]
    right_coords = [right_v_max, right_v_min, right_u_max, right_u_min]
    mouth_coords = [mouth_v_max, mouth_v_min, mouth_u_max, mouth_u_min]

    split_data = {
        'left_coords':left_coords,
        'right_coords':right_coords,
        'mouth_coords':mouth_coords
    }
    return split_data

def decide_mode(video_dir, exp):
    if video_dir == 'video_001':
        if exp == 'contempt':
            return 'weak_test'
        else:
            return 'train'
    elif video_dir == 'video_002':
        if exp == 'sad':
            return 'val'
        else:
            return 'train'
    elif video_dir == 'video_003':
        if exp == 'surprised':
            return 'val'
        else:
            return 'train'
    elif video_dir == 'video_013':
        if exp == 'happy':
            return 'strong_test'
        else:
            return None
    elif video_dir == 'video_018':
        if exp == 'disgusted':
            return 'strong_test'
        else:
            return None
    elif video_dir == 'video_021':
        if exp == 'fear':
            return 'weak_test'
        else:
            return 'train'
    else:
        return 'train'