# coding: utf-8

__author__ = 'cleardusk'

import argparse
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from copy import deepcopy
from utils.tddfa_util import str2bool
from tqdm import tqdm
from fdc_utils import *
from PIL import Image
import random

def main(args):
    video_list = ['video_001', 'video_002', 'video_003', 'video_004', 'video_006', 'video_009', 'video_010', 'video_011', 'video_013', 'video_014', 'video_018', 'video_021', 'video_023', 'video_049']

    # How many target frames we want to capture for each ref frame
    temporal_num = 1 # Hyperparam
    
    # How many frames we skip when capturing the ref frames from the raw video
    # This has been decided by the frame image generation code
    frame_interval = 1

    img_suffix = '.jpg'
    suffix_len = len(img_suffix)

    expression_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']  
    level_list = ['level_1', 'level_2', 'level_3']
    # temporal_num_list = np.arange(temporal_num)+1 # The temporal num list index starts from 1 (only when ref!=gt)
    temporal_num_list = np.arange(temporal_num)
    str_temporal_num_list = [str(a) for a in temporal_num_list]

    ref_angle = 'front'
    left_angle = 'left_60'
    right_angle = 'right_60'
    top_angle = 'top'
    
    for video_dir in video_list:
        # ref_frame_path = os.path.join(args.video_root_path, video_dir, 'front', 'neutral', 'level_1', '001', '000.jpg')
        # print(ref_frame_path)
        # input()
        for exp in expression_list:
            # if exp == 'neutral':
            #     level_list = ['level_1']
            # else:
            #     level_list = ['level_1', 'level_2', 'level_3']
            data_mode = decide_mode(video_dir, exp)
            if data_mode is None: # Skip the data in the folder that is used to generate strong test
                continue
            split_file = os.path.join(args.video_save_path, data_mode+'.txt')
            with open(split_file, 'a') as f:
                for level in level_list:
                    # Parent path for saving data
                    frame_save_parent = os.path.join(args.video_save_path, data_mode, video_dir, exp, level)

                    # Parent path for reading data
                    video_path_parent = os.path.join(args.video_root_path, video_dir, ref_angle, exp, level)

                    ref_path_parent = os.path.join(args.video_root_path, video_dir, 'front', 'neutral', 'level_1')

                    video_clip_list = next(os.walk(video_path_parent))[1]
                    
                    ref_video_clip_list = next(os.walk(ref_path_parent))[1]
                    ref_video_clip_len = len(ref_video_clip_list)

                    # raw_video_clip_list = deepcopy(video_clip_list)
                    # raw_video_clip_len = len(raw_video_clip_list)

                    video_clip_list = [video_clip_list[0]] # Only extract data from the first video clip
                    for clip in video_clip_list:
                        frame_save_dir = os.path.join(frame_save_parent, clip)
                        frame_dir = os.path.join(video_path_parent, clip)
                        
                        # Sample video clip from the ref video folder
                        sample_num = random.randint(0, ref_video_clip_len-1) #randint(a, b) returns an integer in the range [a, b], instead of [a, b-1] 
                        sampled_clip = ref_video_clip_list[sample_num]
                        ref_dir = os.path.join(ref_path_parent, sampled_clip)
                        ref_frames = [f for f in listdir(ref_dir) if isfile(join(ref_dir, f))]
                        ref_frames_len = len(ref_frames)

                        frames = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))] # Read files , not folders
                        frames.sort() # Sort the frames according to the frame number such that the frame in for loop can be extracted in order
                        total_len_frames = len(frames)
                        assert total_len_frames >= temporal_num+1, 'Frames are not enough to extract temporal ones!' # Check if the number of total frames is large than the number of temporal frames

                        # frames = frames[:-1] # Do not use the last frame as the ref frame
                        # # frames = [frames] # Make the frames as a list

                        for i, frame in tqdm(enumerate(frames)):
                            # ==================================
                            # Skip occluded faces (damaged data)
                            if video_dir == 'video_001':
                                if exp == 'happy':
                                    if level == 'level_1':
                                        if clip == '027':
                                            if i < 4:
                                                continue

                            # Sample frame from the ref video clip folder
                            ref_sample_num = random.randint(0, ref_frames_len-1)
                            ref_sampled_frame = ref_frames[ref_sample_num]
                            ref_frame_path = os.path.join(ref_dir, ref_sampled_frame)
                            ref_img = cv2.imread(ref_frame_path)
                            ref_img_2d_landmarks = detect_attributes(ref_img, args) # 2D landmarks of the reference image

                            if np.any(ref_img_2d_landmarks<0):
                                print('ref image 2d landmarks: ', ref_img_2d_landmarks)
                                continue

                            new_ref_img = detect_face(ref_img, ref_img_2d_landmarks) # Newly generated reference image
                            new_ref_img_2d_landmarks, ref_yaw, ref_pitch, ref_roll = detect_attributes(new_ref_img, args, is_pose=True)

                            if np.any(new_ref_img_2d_landmarks<0):
                                print('new_ref image 2d landmarks: ', new_ref_img_2d_landmarks)
                                continue

                            ref_split_data = split_face(new_ref_img_2d_landmarks)
                            ref_left_coords = ref_split_data['left_coords']
                            ref_left_v_min = int(ref_left_coords[1])
                            ref_left_v_max = int(ref_left_coords[0])
                            ref_left_u_min = int(ref_left_coords[3])
                            ref_left_u_max = int(ref_left_coords[2])
                            ref_left_v_range = ref_left_v_max - ref_left_v_min
                            ref_left_u_range = ref_left_u_max - ref_left_u_min
                            ref_left_v_mean, ref_left_u_mean = int((ref_left_coords[0]+ref_left_coords[1])/2), int((ref_left_coords[2]+ref_left_coords[3])/2)

                            ref_right_coords = ref_split_data['right_coords']
                            ref_right_v_min = int(ref_right_coords[1])
                            ref_right_v_max = int(ref_right_coords[0])
                            ref_right_u_min = int(ref_right_coords[3])
                            ref_right_u_max = int(ref_right_coords[2])
                            ref_right_v_range = ref_right_v_max - ref_right_v_min
                            ref_right_u_range = ref_right_u_max - ref_right_u_min
                            ref_right_v_mean, ref_right_u_mean = int((ref_right_coords[0]+ref_right_coords[1])/2), int((ref_right_coords[2]+ref_right_coords[3])/2)

                            ref_mouth_coords = ref_split_data['mouth_coords']
                            ref_mouth_v_min = int(ref_mouth_coords[1])
                            ref_mouth_v_max = int(ref_mouth_coords[0])
                            ref_mouth_u_min = int(ref_mouth_coords[3])
                            ref_mouth_u_max = int(ref_mouth_coords[2])
                            ref_mouth_v_range = ref_mouth_v_max - ref_mouth_v_min
                            ref_mouth_u_range = ref_mouth_u_max - ref_mouth_u_min
                            ref_mouth_v_mean, ref_mouth_u_mean = int((ref_mouth_coords[0]+ref_mouth_coords[1])/2), int((ref_mouth_coords[2]+ref_mouth_coords[3])/2)
                            # ==================================

                            ref_frame_num = int(frame[:-suffix_len]) # Convert the ref frame name to the frame number, i.e., remove the image suffix

                            # Save the reference image into this folder
                            target_save_path = os.path.join(frame_save_dir, str(ref_frame_num).zfill(3)) # Use the ref frame number as the name of the folder name
                            
                            # ==================================
                            ref_frame = 'ref_'+frame
                            
                            # # ref_save_path = os.path.join(target_save_path, ref_frame) # Save new ref image in the new place
                            # if not os.path.exists(target_save_path):
                            #     print('Creating target save path', target_save_path)
                            #     os.makedirs(target_save_path)
                            
                            delta = 30 # Hyperparam. Margin to crop the local area

                            mask = -1*np.ones((new_ref_img.shape[0], new_ref_img.shape[1]))
                            # # Copy the original reference image
                            overlaid_img = deepcopy(new_ref_img)
                            # ==================================
                            for target in str_temporal_num_list: # Loop over the next few target frames
                                target_individual_save_path = os.path.join(target_save_path, target)
                                
                                # ==================================
                                target_frame_number = ref_frame_num + int(target)*frame_interval # Compute the target frame number
                                target_frame_str = str(target_frame_number).zfill(3) + img_suffix # Obtain the target frame name
                                target_frame_path = os.path.join(frame_dir, target_frame_str) # Obtain the target frame path
                                
                                target_gt_img = cv2.imread(target_frame_path) 
                                target_gt_img_2d_landmarks, target_gt_yaw, target_gt_pitch, target_gt_roll = detect_attributes(target_gt_img, args, is_pose=True)

                                # Move to next target image if the angles of the target face deviate too much from the angles of the ref image
                                if np.abs(target_gt_yaw-ref_yaw)>10 or np.abs(target_gt_pitch-ref_pitch)>10 or np.abs(target_gt_roll-ref_roll)>10:
                                    print('Angles differ too much.')
                                    continue

                                if target_gt_img_2d_landmarks is None or np.any(target_gt_img_2d_landmarks<0):
                                    print('target gt image 2d landmarks: ', target_gt_img_2d_landmarks)
                                    continue

                                new_target_gt_img = detect_face(target_gt_img, target_gt_img_2d_landmarks) # Newly generated target gt image
                                new_target_gt_img_gray = cv2.cvtColor(new_target_gt_img, cv2.COLOR_RGB2GRAY)
                                new_target_gt_2d_landmarks = detect_attributes(new_target_gt_img, args)

                                if new_target_gt_2d_landmarks is None or np.any(new_target_gt_2d_landmarks<0):
                                    print('new target gt image 2d landmarks: ', new_target_gt_2d_landmarks)
                                    continue

                                target_gt_split_data = split_face(new_target_gt_2d_landmarks)
                                target_gt_left_coords = target_gt_split_data['left_coords']
                                target_gt_right_coords = target_gt_split_data['right_coords']
                                target_gt_mouth_coords = target_gt_split_data['mouth_coords']

                                # Extract the left part of the target GT image
                                target_left_rgb = new_target_gt_img[target_gt_left_coords[1]-delta:target_gt_left_coords[0]+delta, target_gt_left_coords[3]-delta:target_gt_left_coords[2]+delta]
                                # target_left_gray = new_target_gt_img_gray[target_gt_left_coords[1]-delta:target_gt_left_coords[0]+delta, target_gt_left_coords[3]-delta:target_gt_left_coords[2]+delta]

                                # target_left_v_range = target_left_rgb.shape[0]
                                # target_left_u_range = target_left_rgb.shape[1]

                                # target_left_v_min = ref_left_v_mean - int(target_left_v_range/2)
                                # target_left_v_max = target_left_v_min + target_left_v_range
                                # target_left_u_min = ref_left_u_mean - int(target_left_u_range/2)
                                # target_left_u_max = target_left_u_min + target_left_u_range

                                # repeat_target_left_gray = np.repeat(target_left_gray[:,:,np.newaxis], 3, axis=2)

                                # Extract the right part of the target gt image
                                target_right_rgb = new_target_gt_img[target_gt_right_coords[1]-delta:target_gt_right_coords[0]+delta, target_gt_right_coords[3]-delta:target_gt_right_coords[2]+delta]
                                # target_right_gray = new_target_gt_img_gray[target_gt_right_coords[1]-delta:target_gt_right_coords[0]+delta, target_gt_right_coords[3]-delta:target_gt_right_coords[2]+delta]

                                # target_right_v_range = target_right_rgb.shape[0]
                                # target_right_u_range = target_right_rgb.shape[1]

                                # target_right_v_min = ref_right_v_mean - int(target_right_v_range/2)
                                # target_right_v_max = target_right_v_min + target_right_v_range
                                # target_right_u_min = ref_right_u_mean - int(target_right_u_range/2)
                                # target_right_u_max = target_right_u_min + target_right_u_range

                                # repeat_target_right_gray = np.repeat(target_right_gray[:,:,np.newaxis], 3, axis=2)

                                # Extract the mouth part of the target gt image
                                target_mouth_rgb = new_target_gt_img[target_gt_mouth_coords[1]-int(2*delta):target_gt_mouth_coords[0]+int(4.3*delta), target_gt_mouth_coords[3]-int(3.5*delta):target_gt_mouth_coords[2]+int(3.5*delta)]
                                # target_mouth_gray = new_target_gt_img_gray[target_gt_mouth_coords[1]-delta:target_gt_mouth_coords[0]+delta, target_gt_mouth_coords[3]-delta:target_gt_mouth_coords[2]+delta]

                                # target_mouth_v_range = target_mouth_rgb.shape[0]
                                # target_mouth_u_range = target_mouth_rgb.shape[1]

                                # target_mouth_v_min = ref_mouth_v_mean - int(target_mouth_v_range/2)
                                # target_mouth_v_max = target_mouth_v_min + target_mouth_v_range
                                # target_mouth_u_min = ref_mouth_u_mean - int(target_mouth_u_range/2)
                                # target_mouth_u_max = target_mouth_u_min + target_mouth_u_range

                                # repeat_target_mouth_gray = np.repeat(target_mouth_gray[:,:,np.newaxis], 3, axis=2)

                                # Extract left target image
                                left_video_path_parent = os.path.join(args.video_root_path, video_dir, left_angle, exp, level) # Level folder
                                left_frame_dir = os.path.join(left_video_path_parent, clip) # Clip folder
                                left_target_frame_path = os.path.join(left_frame_dir, target_frame_str) # Target frame 
                                target_left_img = cv2.imread(left_target_frame_path) # Read by OpenCV
                                target_left_img_2d_landmarks = detect_attributes(target_left_img, args)
                                if target_left_img_2d_landmarks is None: # It is possible to not detect landmarks from the left image
                                    continue
                                
                                left_split_data = split_face(target_left_img_2d_landmarks)
                                left_coords = left_split_data['left_coords']
                                
                                normalized_left_eyebrow = left_split_data['normalized_left_eyebrow']
                                normalized_left_eye = left_split_data['normalized_left_eye']

                                target_left_img_gray = cv2.cvtColor(target_left_img, cv2.COLOR_RGB2GRAY)
                                left_left = target_left_img_gray[left_coords[1]-delta:left_coords[0]+delta, left_coords[3]-delta:left_coords[2]+delta]
                                left_left_img = Image.fromarray(left_left)
                                target_left_rgb_img = Image.fromarray(target_left_rgb) # Left gt

                                resized_left_left_img = left_left_img.resize((ref_left_u_range+2*delta, ref_left_v_range+2*delta), Image.BICUBIC)
                                resized_target_left_rgb_img = target_left_rgb_img.resize((ref_left_u_range+2*delta, ref_left_v_range+2*delta), Image.BICUBIC)

                                left_left = np.asarray(resized_left_left_img)
                                target_left_rgb = np.asarray(resized_target_left_rgb_img)

                                repeat_left_left = np.repeat(left_left[:,:,np.newaxis], 3, axis=2)
                                overlaid_img[ref_left_v_min-delta:ref_left_v_max+delta, ref_left_u_min-delta:ref_left_u_max+delta, :] = repeat_left_left
                                mask[ref_left_v_min-delta:ref_left_v_max+delta, ref_left_u_min-delta:ref_left_u_max+delta] = 1

                                # Extract right target image
                                right_video_path_parent = os.path.join(args.video_root_path, video_dir, right_angle, exp, level)
                                right_frame_dir = os.path.join(right_video_path_parent, clip)
                                right_target_frame_path = os.path.join(right_frame_dir, target_frame_str)
                                target_right_img = cv2.imread(right_target_frame_path)
                                target_right_img_2d_landmarks = detect_attributes(target_right_img, args)
                                if target_right_img_2d_landmarks is None: # It is possible to not detect landmarks from the right image
                                    continue

                                right_split_data = split_face(target_right_img_2d_landmarks)
                                right_coords = right_split_data['right_coords']

                                normalized_right_eyebrow = right_split_data['normalized_right_eyebrow']
                                normalized_right_eye = right_split_data['normalized_right_eye']

                                target_right_img_gray = cv2.cvtColor(target_right_img, cv2.COLOR_RGB2GRAY)
                                right_right = target_right_img_gray[right_coords[1]-delta:right_coords[0]+delta, right_coords[3]-delta:right_coords[2]+delta]
                                right_right_rgb = target_right_img[right_coords[1]-delta:right_coords[0]+delta, right_coords[3]-delta:right_coords[2]+delta]
                                right_right_img = Image.fromarray(right_right)
                                target_right_rgb_img = Image.fromarray(target_right_rgb)

                                resized_right_right_img = right_right_img.resize((ref_right_u_range+2*delta, ref_right_v_range+2*delta), Image.BICUBIC)
                                resized_target_right_rgb_img = target_right_rgb_img.resize((ref_right_u_range+2*delta, ref_right_v_range+2*delta), Image.BICUBIC)

                                right_right = np.asarray(resized_right_right_img)
                                target_right_rgb = np.asarray(resized_target_right_rgb_img)
                                repeat_right_right = np.repeat(right_right[:,:,np.newaxis], 3, axis=2)
                                overlaid_img[ref_right_v_min-delta:ref_right_v_max+delta, ref_right_u_min-delta:ref_right_u_max+delta, :] = repeat_right_right
                                mask[ref_right_v_min-delta:ref_right_v_max+delta, ref_right_u_min-delta:ref_right_u_max+delta] = 2

                                # Extract top target image
                                top_video_path_parent = os.path.join(args.video_root_path, video_dir, top_angle, exp, level)
                                top_frame_dir = os.path.join(top_video_path_parent, clip)
                                top_target_frame_path = os.path.join(top_frame_dir, target_frame_str)
                                target_top_img = cv2.imread(top_target_frame_path)
                                target_top_img_2d_landmarks = detect_attributes(target_top_img, args)
                                
                                top_split_data = split_face(target_top_img_2d_landmarks)
                                top_coords = top_split_data['mouth_coords']

                                normalized_mouth = top_split_data['normalized_mouth']

                                target_top_img_gray = cv2.cvtColor(target_top_img, cv2.COLOR_RGB2GRAY)
                                
                                # # 111
                                # top_mouth = target_top_img_gray[top_coords[1]-delta:top_coords[0]+delta, top_coords[3]-delta:top_coords[2]+delta]
                                top_mouth_u_min = top_coords[3]-3*delta
                                top_mouth_u_max = top_coords[2]+3*delta
                                top_mouth_v_min = top_coords[1]-2*delta
                                # top_mouth_v_max = target_top_img_gray.shape[0]
                                top_mouth_v_max = top_coords[0]+2*delta

                                top_mouth = target_top_img_gray[top_mouth_v_min:top_mouth_v_max, top_mouth_u_min:top_mouth_u_max]

                                top_mouth_img = Image.fromarray(top_mouth)
                                target_mouth_rgb_img = Image.fromarray(target_mouth_rgb)
                                
                                # Enlarge the mouth area from the orignal one
                                # The mouth patch is extacted from the raw top image
                                # While the ref image is cropped
                                # So there should be a linear transformation to fit the mouth patch into the ref mask
                                ref_mouth_u_min = int(ref_mouth_u_min - 4.5*delta)
                                ref_mouth_u_max = int(ref_mouth_u_max + 4.3*delta)
                                ref_mouth_v_min = int(ref_mouth_v_min - 2.5*delta)
                                ref_mouth_v_max = int(ref_mouth_v_max + 3.5*delta)

                                ref_mouth_u_range = ref_mouth_u_max - ref_mouth_u_min
                                ref_mouth_v_range = ref_mouth_v_max - ref_mouth_v_min

                                # # 111
                                # resized_top_mouth_img = top_mouth_img.resize((ref_mouth_u_range+2*delta, ref_mouth_v_range+2*delta), Image.BICUBIC)
                                # resized_target_mouth_rgb_img = target_mouth_rgb_img.resize((ref_mouth_u_range+2*delta, ref_mouth_v_range+2*delta), Image.BICUBIC)

                                resized_top_mouth_img = top_mouth_img.resize((ref_mouth_u_range, ref_mouth_v_range), Image.BICUBIC)
                                resized_target_mouth_rgb_img = target_mouth_rgb_img.resize((ref_mouth_u_range, ref_mouth_v_range), Image.BICUBIC)
                                
                                top_mouth = np.asarray(resized_top_mouth_img)
                                target_mouth_rgb = np.asarray(resized_target_mouth_rgb_img)
                                repeat_top_mouth = np.repeat(top_mouth[:,:,np.newaxis], 3, axis=2)

                                # # 111
                                # overlaid_img[ref_mouth_v_min-delta:ref_mouth_v_max+delta, ref_mouth_u_min-delta:ref_mouth_u_max+delta, :] = repeat_top_mouth

                                # TODO: identify the problem: ValueError: could not broadcast input array from shape (297,444,3) into shape (296,444,3)
                                try:
                                    overlaid_img[ref_mouth_v_min:ref_mouth_v_max, ref_mouth_u_min:ref_mouth_u_max, :] = repeat_top_mouth
                                except:
                                    continue
                                
                                # # 111
                                # mask[ref_mouth_v_min-delta:ref_mouth_v_max+delta, ref_mouth_u_min-delta:ref_mouth_u_max+delta] = 3

                                mask[ref_mouth_v_min:ref_mouth_v_max, ref_mouth_u_min:ref_mouth_u_max] = 3

                                # Create save path for data
                                ref_save_path = os.path.join(target_individual_save_path, 'reference.png') # Save new ref image in the new place
                                overlaid_save_path = os.path.join(target_individual_save_path, 'overlaid.png')
                                mask_save_path = os.path.join(target_individual_save_path, 'mask.npy')
                                
                                normalized_left_eyebrow_save_path = os.path.join(target_individual_save_path, 'normalized_left_eyebrow.npy')
                                normalized_left_eye_save_path = os.path.join(target_individual_save_path, 'normalized_left_eye.npy')
                                normalized_right_eyebrow_save_path = os.path.join(target_individual_save_path, 'normalized_right_eyebrow.npy')
                                normalized_right_eye_save_path = os.path.join(target_individual_save_path, 'normalized_right_eye.npy')
                                normalized_mouth_save_path = os.path.join(target_individual_save_path, 'normalized_mouth.npy')

                                left_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'left_patch_gray_angle.png')
                                right_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'right_patch_gray_angle.png')
                                mouth_patch_gray_angle_save_path = os.path.join(target_individual_save_path, 'mouth_patch_gray_angle.png')
                                left_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'left_patch_rgb_gt.png')
                                right_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'right_patch_rgb_gt.png')
                                mouth_patch_rgb_gt_save_path = os.path.join(target_individual_save_path, 'mouth_patch_rgb_gt.png')
                                target_gt_save_path = os.path.join(target_individual_save_path, 'gt.png')

                                if not os.path.exists(target_individual_save_path): # Create a target individual folder if the folder does not exist
                                    print('Creating target individual folders...')
                                    os.makedirs(target_individual_save_path)

                                cv2.imwrite(target_gt_save_path, new_target_gt_img)

                                # # Save newly cropped reference image
                                cv2.imwrite(ref_save_path, new_ref_img) 

                                # Save overlaid image
                                cv2.imwrite(overlaid_save_path, overlaid_img)
                                
                                # Save mask and landmarks
                                np.save(mask_save_path, mask)
                                np.save(normalized_left_eyebrow_save_path, normalized_left_eyebrow)
                                np.save(normalized_left_eye_save_path, normalized_left_eye)
                                np.save(normalized_right_eyebrow_save_path, normalized_right_eyebrow)
                                np.save(normalized_right_eye_save_path, normalized_right_eye)
                                np.save(normalized_mouth_save_path, normalized_mouth)

                                # Save grayscale patches with angles
                                cv2.imwrite(left_patch_gray_angle_save_path, repeat_left_left)
                                cv2.imwrite(right_patch_gray_angle_save_path, repeat_right_right)
                                cv2.imwrite(mouth_patch_gray_angle_save_path, repeat_top_mouth)

                                # Save rgb gt patches
                                cv2.imwrite(left_patch_rgb_gt_save_path, target_left_rgb)
                                cv2.imwrite(right_patch_rgb_gt_save_path, target_right_rgb)
                                cv2.imwrite(mouth_patch_rgb_gt_save_path, target_mouth_rgb)
                                # ==================================
                                f.write(target_individual_save_path)
                                f.write('\n')
                            print('Finish one ref frame: ', type, frame)
                f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-r', '--video_root_path', type=str, default='/home/uss00067/Datasets/New_frames_1')
    parser.add_argument('-s', '--video_save_path', type=str, default='/home/uss00067/Datasets/FDC_7')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
