# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from copy import deepcopy

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
from utils.tddfa_util import str2bool

from tqdm import tqdm
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
        sys.exit(-1)

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
    # u_size = 700
    # if is_debug:
    #     print('In detect_face, shape of input img: ', img.shape)

    # u_coords = landmark_raw[0,:]
    # v_coords = landmark_raw[1,:]
    # u_max = int(np.max(u_coords))
    # u_min = int(np.min(u_coords))
    # v_max = int(np.max(v_coords))
    # v_min = int(np.min(v_coords))

    # middle_u = int((u_max+u_min)/2)
    # middle_v = int((v_max+v_min)/2)

    # print('original image shape: ', img.shape)

    # new_v_min = middle_v - int(v_size/2)
    # new_v_max = new_v_min + v_size
    # assert new_v_max < img.shape[0], 'new_v_max exceeds the boundary'
    
    # new_u_min = middle_u - int(u_size/2)
    # new_u_max = new_u_min + u_size
    # print(new_u_min, new_u_max)
    # assert new_u_max < img.shape[1], 'new_u_max exceeds the boundary'

    # if new_u_max < 0 or new_u_min < 0 or new_v_max < 0 or new_v_min < 0:
    #     print('Index: ', new_u_max, new_u_min, new_v_max, new_v_min)
    #     raise RuntimeError('Index is negative!')

    # new_img = img[new_v_min:new_v_max, new_u_min:new_u_max, :]
    # print('In detect_face, output shape: ', new_img.shape)
    # # input()

    new_img = img[:,420:1500,:] # Crop the image to [1080, 1080]
    scale_factor = 0.7
    new_img = Image.fromarray(new_img) # Convert numpy image to PIL image
    new_img = recrop_img(new_img, scale_factor)

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

def main(args):
    video_dir_list = ['video_001'] # Which video folder we are working on

    # How many target frames we want to capture for each ref frame
    temporal_num = 5 # Hyperparam
    
    # How many frames we skip when capturing the ref frames from the raw video
    # This has been decided by the frame image generation code
    frame_interval = 5 

    img_suffix = '.jpg'
    suffix_len = len(img_suffix)

    # angle_list = ['front', 'left_60', 'right_60', 'top']
    expression_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']  
    level_list = ['level_1', 'level_2', 'level_3']
    temporal_num_list = np.arange(temporal_num)+1 # The temporal num list index starts from 1
    str_temporal_num_list = [str(a) for a in temporal_num_list]

    ref_angle = 'front'
    left_angle = 'left_60'
    right_angle = 'right_60'
    top_angle = 'top'

    for video_dir in video_dir_list:
        for exp in expression_list:
            if exp == 'neutral':
                level_list = ['level_1']
            for level in level_list:
                frame_save_parent = os.path.join(args.video_save_path, video_dir, exp, level)

                video_path_parent = os.path.join(args.video_root_path, video_dir, ref_angle, exp, level)
                video_clip_list = next(os.walk(video_path_parent))[1]
                for clip in video_clip_list:
                    frame_save_dir = os.path.join(frame_save_parent, clip)
                    frame_dir = os.path.join(video_path_parent, clip)
                    frames = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))] # Read files , not folders
                    frames.sort() # Sort the frames according to the frame number such that the frame in for loop can be extracted in order
                    total_len_frames = len(frames)
                    assert total_len_frames >= temporal_num+1, 'Frames are not enough to extract temporal ones!' # Check if the number of total frames is large than the number of temporal frames

                    # for frame in frames:
                    for i, frame in tqdm(enumerate(frames)):
                        # frame = frames[i]
                        ref_frame_path = os.path.join(frame_dir, frame) # Reference image
                        ref_img = cv2.imread(ref_frame_path)
                        ref_img_2d_landmarks = detect_attributes(ref_img, args) # 2D landmarks of the reference image

                        if np.any(ref_img_2d_landmarks<0):
                            print('ref image 2d landmarks: ', ref_img_2d_landmarks)
                        assert not np.any(ref_img_2d_landmarks<0), 'ref_img_2d_landmarks have negative values!'

                        new_ref_img = detect_face(ref_img, ref_img_2d_landmarks) # Newly generated reference image
                        new_ref_img_2d_landmarks = detect_attributes(new_ref_img, args)

                        if np.any(new_ref_img_2d_landmarks<0):
                            print('new_ref image 2d landmarks: ', new_ref_img_2d_landmarks)
                        assert not np.any(new_ref_img_2d_landmarks<0), 'new_ref_img_2d_landmarks have negative values!'

                        ref_split_data = split_face(new_ref_img_2d_landmarks)
                        
                        ref_left_coords = ref_split_data['left_coords']
                        ref_left_v_mean, ref_left_u_mean = int((ref_left_coords[0]+ref_left_coords[1])/2), int((ref_left_coords[2]+ref_left_coords[3])/2)

                        ref_right_coords = ref_split_data['right_coords']
                        ref_right_v_mean, ref_right_u_mean = int((ref_right_coords[0]+ref_right_coords[1])/2), int((ref_right_coords[2]+ref_right_coords[3])/2)

                        ref_mouth_coords = ref_split_data['mouth_coords']
                        ref_mouth_v_mean, ref_mouth_u_mean = int((ref_mouth_coords[0]+ref_mouth_coords[1])/2), int((ref_mouth_coords[2]+ref_mouth_coords[3])/2)

                        ref_frame_num = int(frame[:-suffix_len]) # Convert the ref frame name to the frame number, i.e., remove the image suffix

                        # Save the reference image into this folder
                        target_save_path = os.path.join(frame_save_dir, str(ref_frame_num).zfill(3)) # Use the ref frame number as the name of the folder name
                        ref_frame = 'ref_'+frame
                        
                        ref_save_path = os.path.join(target_save_path, ref_frame) # Save new ref image in the new place
                        if not os.path.exists(target_save_path):
                            print('Save ref images to ...', ref_save_path)
                            os.makedirs(target_save_path)
                        cv2.imwrite(ref_save_path, new_ref_img)
                        
                        delta = 30 # Hyperparam. Margin to crop the local area
                        
                        # Copy the original reference image
                        overlaid_img = deepcopy(new_ref_img)
                        mask = -1*np.ones((overlaid_img.shape[0], overlaid_img.shape[1]))

                        overlaid_img_1 = deepcopy(new_ref_img)
                        overlaid_img_2 = deepcopy(new_ref_img)
                        overlaid_img_3 = deepcopy(new_ref_img)

                        is_partial_data = False
                        if is_partial_data:
                            overlaid_img_4 = deepcopy(new_ref_img)
                            overlaid_img_5 = deepcopy(new_ref_img)
                            mask_4 = -1*np.ones((overlaid_img.shape[0], overlaid_img.shape[1]))

                        for target in str_temporal_num_list: # Loop over the next few target frames
                            target_frame_number = ref_frame_num + int(target)*frame_interval # Compute the target frame number
                            target_frame_str = str(target_frame_number).zfill(3) + img_suffix # Obtaint the target frame name
                            target_frame_path = os.path.join(frame_dir, target_frame_str) # Obtain the target frame path
                            
                            target_gt_img = cv2.imread(target_frame_path) 
                            target_gt_img_2d_landmarks = detect_attributes(target_gt_img, args)
                            
                            if np.any(target_gt_img_2d_landmarks<0):
                                print('target gt image 2d landmarks: ', target_gt_img_2d_landmarks)
                            assert not np.any(target_gt_img_2d_landmarks<0), 'target_gt_img_2d_landmarks have negative values!'

                            new_target_gt_img = detect_face(target_gt_img, target_gt_img_2d_landmarks) # Newly generated target gt image
                            new_target_gt_img_gray = cv2.cvtColor(new_target_gt_img, cv2.COLOR_RGB2GRAY)
                            new_target_gt_2d_landmarks = detect_attributes(new_target_gt_img, args)

                            target_gt_split_data = split_face(new_target_gt_2d_landmarks)
                            target_gt_left_coords = target_gt_split_data['left_coords']
                            target_gt_right_coords = target_gt_split_data['right_coords']
                            target_gt_mouth_coords = target_gt_split_data['mouth_coords']

                            # Extract the left part of the target gt image
                            target_left_rgb = new_target_gt_img[target_gt_left_coords[1]-delta:target_gt_left_coords[0]+delta, target_gt_left_coords[3]-delta:target_gt_left_coords[2]+delta]
                            target_left_gray = new_target_gt_img_gray[target_gt_left_coords[1]-delta:target_gt_left_coords[0]+delta, target_gt_left_coords[3]-delta:target_gt_left_coords[2]+delta]

                            target_left_v_range = target_left_rgb.shape[0]
                            target_left_u_range = target_left_rgb.shape[1]

                            target_left_v_min = ref_left_v_mean - int(target_left_v_range/2)
                            target_left_v_max = target_left_v_min + target_left_v_range
                            target_left_u_min = ref_left_u_mean - int(target_left_u_range/2)
                            target_left_u_max = target_left_u_min + target_left_u_range

                            repeat_target_left_gray = np.repeat(target_left_gray[:,:,np.newaxis], 3, axis=2)

                            overlaid_img_2[target_left_v_min:target_left_v_max, target_left_u_min:target_left_u_max, :] = repeat_target_left_gray
                            overlaid_img_3[target_left_v_min:target_left_v_max, target_left_u_min:target_left_u_max, :] = target_left_rgb

                            # Extract the right part of the target gt image
                            target_right_rgb = new_target_gt_img[target_gt_right_coords[1]-delta:target_gt_right_coords[0]+delta, target_gt_right_coords[3]-delta:target_gt_right_coords[2]+delta]
                            target_right_gray = new_target_gt_img_gray[target_gt_right_coords[1]-delta:target_gt_right_coords[0]+delta, target_gt_right_coords[3]-delta:target_gt_right_coords[2]+delta]

                            target_right_v_range = target_right_rgb.shape[0]
                            target_right_u_range = target_right_rgb.shape[1]

                            target_right_v_min = ref_right_v_mean - int(target_right_v_range/2)
                            target_right_v_max = target_right_v_min + target_right_v_range
                            target_right_u_min = ref_right_u_mean - int(target_right_u_range/2)
                            target_right_u_max = target_right_u_min + target_right_u_range

                            repeat_target_right_gray = np.repeat(target_right_gray[:,:,np.newaxis], 3, axis=2)

                            overlaid_img_2[target_right_v_min:target_right_v_max, target_right_u_min:target_right_u_max, :] = repeat_target_right_gray
                            overlaid_img_3[target_right_v_min:target_right_v_max, target_right_u_min:target_right_u_max, :] = target_right_rgb

                            # Extract the mouth part of the target gt image
                            target_mouth_rgb = new_target_gt_img[target_gt_mouth_coords[1]-delta:target_gt_mouth_coords[0]+delta, target_gt_mouth_coords[3]-delta:target_gt_mouth_coords[2]+delta]
                            target_mouth_gray = new_target_gt_img_gray[target_gt_mouth_coords[1]-delta:target_gt_mouth_coords[0]+delta, target_gt_mouth_coords[3]-delta:target_gt_mouth_coords[2]+delta]

                            target_mouth_v_range = target_mouth_rgb.shape[0]
                            target_mouth_u_range = target_mouth_rgb.shape[1]

                            target_mouth_v_min = ref_mouth_v_mean - int(target_mouth_v_range/2)
                            target_mouth_v_max = target_mouth_v_min + target_mouth_v_range
                            target_mouth_u_min = ref_mouth_u_mean - int(target_mouth_u_range/2)
                            target_mouth_u_max = target_mouth_u_min + target_mouth_u_range

                            repeat_target_mouth_gray = np.repeat(target_mouth_gray[:,:,np.newaxis], 3, axis=2)

                            overlaid_img_2[target_mouth_v_min:target_mouth_v_max, target_mouth_u_min:target_mouth_u_max, :] = repeat_target_mouth_gray
                            overlaid_img_3[target_mouth_v_min:target_mouth_v_max, target_mouth_u_min:target_mouth_u_max, :] = target_mouth_rgb
                            
                            if is_partial_data:
                                overlaid_img_4[target_mouth_v_min:target_mouth_v_max, target_mouth_u_min:target_mouth_u_max, :] = repeat_target_mouth_gray
                                overlaid_img_5[target_mouth_v_min:target_mouth_v_max, target_mouth_u_min:target_mouth_u_max, :] = target_mouth_rgb

                            target_individual_save_path = os.path.join(target_save_path, target)
                            if not os.path.exists(target_individual_save_path): # Create a target individual folder if the folder does not exist
                                print('Creating target individual folders...')
                                os.makedirs(target_individual_save_path)

                            target_gt_save_path = os.path.join(target_individual_save_path, target_frame_str)
                            cv2.imwrite(target_gt_save_path, new_target_gt_img)

                            # Extract left target image
                            left_video_path_parent = os.path.join(args.video_root_path, video_dir, left_angle, exp, level) # Level folder
                            left_frame_dir = os.path.join(left_video_path_parent, clip) # Clip folder
                            left_target_frame_path = os.path.join(left_frame_dir, target_frame_str) # Target frame 
                            target_left_img = cv2.imread(left_target_frame_path) # Read by OpenCV
                            target_left_img_2d_landmarks = detect_attributes(target_left_img, args)
                            
                            if np.any(target_left_img_2d_landmarks<0):
                                print('target left image 2d landmarks: ', target_left_img_2d_landmarks)
                            assert not np.any(target_left_img_2d_landmarks<0), 'target_left_img_2d_landmarks have negative values!'

                            new_target_left_img = detect_face(target_left_img, target_left_img_2d_landmarks) # Newly generated target left_image
                            new_left_2d_landmarks = detect_attributes(new_target_left_img, args)

                            if np.any(new_left_2d_landmarks<0):
                                print('Left 2d landmarks: ', new_left_2d_landmarks)
                            assert not np.any(new_left_2d_landmarks<0), 'new_left_2d_landmarks have negative values!'

                            new_target_left_img_gray = cv2.cvtColor(new_target_left_img, cv2.COLOR_RGB2GRAY)

                            left_split_data = split_face(new_left_2d_landmarks)
                            # Left eye of left angle image
                            left_coords = left_split_data['left_coords']

                            # Add the margin when crop the local attribute
                            left_left = new_target_left_img_gray[left_coords[1]-delta:left_coords[0]+delta, left_coords[3]-delta:left_coords[2]+delta]
                            left_left_rgb = new_target_left_img[left_coords[1]-delta:left_coords[0]+delta, left_coords[3]-delta:left_coords[2]+delta]

                            left_v_range = left_left.shape[0]
                            left_u_range = left_left.shape[1]

                            left_v_min = ref_left_v_mean - int(left_v_range/2)
                            # left_v_max = ref_left_v_mean + int(left_v_range/2) # This can cause dimension conflict!
                            left_v_max = left_v_min + left_v_range

                            left_u_min = ref_left_u_mean - int(left_u_range/2)
                            left_u_max = left_u_min + left_u_range

                            repeat_left_left = np.repeat(left_left[:,:,np.newaxis], 3, axis=2)
                            # Replace the left eye
                            overlaid_img[left_v_min:left_v_max, left_u_min:left_u_max, :] = repeat_left_left
                            mask[left_v_min:left_v_max, left_u_min:left_u_max] = 1

                            overlaid_img_1[left_v_min:left_v_max, left_u_min:left_u_max, :] = left_left_rgb

                            # Extract right target image
                            right_video_path_parent = os.path.join(args.video_root_path, video_dir, right_angle, exp, level)
                            right_frame_dir = os.path.join(right_video_path_parent, clip)
                            right_target_frame_path = os.path.join(right_frame_dir, target_frame_str)
                            target_right_img = cv2.imread(right_target_frame_path)
                            target_right_img_2d_landmarks = detect_attributes(target_right_img, args)
                            
                            if np.any(target_right_img_2d_landmarks<0):
                                print('target right image 2d landmarks: ', target_right_img_2d_landmarks)
                            assert not np.any(target_right_img_2d_landmarks<0), 'target_right_img_2d_landmarks have negative values!'

                            new_target_right_img = detect_face(target_right_img, target_right_img_2d_landmarks)
                            new_target_right_img_gray = cv2.cvtColor(new_target_right_img, cv2.COLOR_RGB2GRAY)

                            new_right_2d_landmarks = detect_attributes(new_target_right_img, args)
                            
                            if np.any(new_right_2d_landmarks<0):
                                print('new right 2d landmarks: ', new_right_2d_landmarks)
                            assert not np.any(new_right_2d_landmarks<0), 'new_right_2d_landmarks have negative values!'

                            right_split_data = split_face(new_right_2d_landmarks)

                            right_coords = right_split_data['right_coords']

                            # Right eye of right angle image
                            right_right = new_target_right_img_gray[right_coords[1]-delta:right_coords[0]+delta, right_coords[3]-delta:right_coords[2]+delta]
                            right_right_rgb = new_target_right_img[right_coords[1]-delta:right_coords[0]+delta, right_coords[3]-delta:right_coords[2]+delta]

                            right_v_range, right_u_range = right_right.shape
                            right_v_min = ref_right_v_mean - int(right_v_range/2)
                            right_v_max = right_v_min + right_v_range
                            right_u_min = ref_right_u_mean - int(right_u_range/2)
                            right_u_max = right_u_min + right_u_range
                            
                            repeat_right_right = np.repeat(right_right[:,:,np.newaxis], 3, axis=2)
                            # Replace the right eye
                            overlaid_img[right_v_min:right_v_max, right_u_min:right_u_max, :] = repeat_right_right
                            mask[right_v_min:right_v_max, right_u_min:right_u_max] = 1

                            overlaid_img_1[right_v_min:right_v_max, right_u_min:right_u_max, :] = right_right_rgb

                            # Extract top target image
                            top_video_path_parent = os.path.join(args.video_root_path, video_dir, top_angle, exp, level)
                            top_frame_dir = os.path.join(top_video_path_parent, clip)
                            top_target_frame_path = os.path.join(top_frame_dir, target_frame_str)
                            target_top_img = cv2.imread(top_target_frame_path)
                            target_top_img_2d_landmarks = detect_attributes(target_top_img, args)
                            
                            if np.any(target_top_img_2d_landmarks<0):
                                print('target top image 2d landmarks: ', target_top_img_2d_landmarks)
                            assert not np.any(target_top_img_2d_landmarks<0), 'target_top_img_2d_landmarks have negative values!'

                            new_target_top_img = detect_face(target_top_img, target_top_img_2d_landmarks)
                            new_target_top_img_gray = cv2.cvtColor(new_target_top_img, cv2.COLOR_RGB2GRAY)

                            new_top_2d_landmarks = detect_attributes(new_target_top_img, args)
                            
                            if np.any(new_top_2d_landmarks<0):
                                print('new top 2d landmarks: ', new_top_2d_landmarks)
                            assert not np.any(new_top_2d_landmarks<0), 'new_top_2d_landmarks have negative values!'

                            top_split_data = split_face(new_top_2d_landmarks) 
                            mouth_coords = top_split_data['mouth_coords']
                            top_mouth = new_target_top_img_gray[mouth_coords[1]-delta:mouth_coords[0]+delta, mouth_coords[3]-delta:mouth_coords[2]+delta]
                            top_mouth_rgb = new_target_top_img[mouth_coords[1]-delta:mouth_coords[0]+delta, mouth_coords[3]-delta:mouth_coords[2]+delta]

                            mouth_v_range, mouth_u_range = top_mouth.shape
                            mouth_v_min = ref_mouth_v_mean - int(mouth_v_range/2)
                            mouth_v_max = mouth_v_min + mouth_v_range
                            mouth_u_min = ref_mouth_u_mean - int(mouth_u_range/2)
                            mouth_u_max = mouth_u_min + mouth_u_range

                            repeat_top_mouth = np.repeat(top_mouth[:,:,np.newaxis], 3, axis=2)

                            # Replace the mouth
                            overlaid_img[mouth_v_min:mouth_v_max, mouth_u_min:mouth_u_max, :] = repeat_top_mouth
                            mask[mouth_v_min:mouth_v_max, mouth_u_min:mouth_u_max] = 1

                            if is_partial_data:
                                mask_4[mouth_v_min:mouth_v_max, mouth_u_min:mouth_u_max] = 1

                            overlaid_img_1[mouth_v_min:mouth_v_max, mouth_u_min:mouth_u_max] = top_mouth_rgb
                            
                            ref_path = '/home/uss00067/Datasets/FDC/video_001/angry/level_1/024/000/ref_000.jpg'
                            ref_img = cv2.imread(ref_path)
                            ref_save_path = os.path.join(target_individual_save_path, 'ref_000.png')

                            overlaid_save_path = os.path.join(target_individual_save_path, 'overlaid.png')
                            mask_save_path = os.path.join(target_individual_save_path, 'mask.npy')
                            # overlaid_save_path_1 = os.path.join(target_individual_save_path, 'overlaid_1.png')
                            # overlaid_save_path_2 = os.path.join(target_individual_save_path, 'overlaid_2.png')
                            # overlaid_save_path_3 = os.path.join(target_individual_save_path, 'overlaid_3.png')

                            left_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'left_patch_gray_angle.png')
                            right_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'right_patch_gray_angle.png')
                            mouth_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'mouth_patch_gray_angle.png')
                            
                            left_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'left_patch_rgb_gt.png')
                            right_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'right_patch_rgb_gt.png')
                            mouth_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'mouth_patch_rgb_gt.png')

                            if is_partial_data:
                                overlaid_save_path_4 = os.path.join(target_individual_save_path, 'overlaid_4.jpeg')
                                overlaid_save_path_5 = os.path.join(target_individual_save_path, 'overlaid_5.jpeg')
                                mask_4_save_path = os.path.join(target_individual_save_path, 'mask_4.npy')

                            print('Save path: ', overlaid_save_path)

                            # Save reference image
                            cv2.imwrite(ref_save_path, ref_img)

                            # Save overlaid image
                            cv2.imwrite(overlaid_save_path, overlaid_img)
                            np.save(mask_save_path, mask)
                            # cv2.imwrite(overlaid_save_path_1, overlaid_img_1)
                            # cv2.imwrite(overlaid_save_path_2, overlaid_img_2)
                            # cv2.imwrite(overlaid_save_path_3, overlaid_img_3)

                            # Save grayscale patches with angles
                            cv2.imwrite(left_patch_gray_angle_save_path, repeat_left_left)
                            cv2.imwrite(right_patch_gray_angle_save_path, repeat_right_right)
                            cv2.imwrite(mouth_patch_gray_angle_save_path, repeat_top_mouth)

                            # Save rgb gt patches
                            cv2.imwrite(left_patch_rgb_gt_save_path, target_left_rgb)
                            cv2.imwrite(right_patch_rgb_gt_save_path, target_right_rgb)
                            cv2.imwrite(mouth_patch_rgb_gt_save_path, target_mouth_rgb)

                            if is_partial_data:
                                cv2.imwrite(overlaid_save_path_4, overlaid_img_4)
                                cv2.imwrite(overlaid_save_path_5, overlaid_img_5)
                                np.save(mask_4_save_path, mask_4)
                            
                        print('Finish one ref frame')
                        input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-r', '--video_root_path', type=str, default='/home/uss00067/Datasets/MEAD_Frames')
    parser.add_argument('-s', '--video_save_path', type=str, default='/home/uss00067/Datasets/FDC')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
