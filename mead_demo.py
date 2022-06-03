# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import numpy as np
from os import listdir
from os.path import isfile, join

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

def detect_attributes(img, attribute_type):
    
def main(args):
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

    # Generate target images and overlaid images

    video_dir_root = ['video_1'] # Which video folder we are working on

    temporal_num = 3 # How many target frames we want to capture for each ref frame
    frame_interval = 5 # How many frames we skip when capturing the ref frames from the raw video
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

    for exp in expression_list:
        if exp == 'neutral':
            level_list = ['level_1']
        for level in level_list:
            frame_save_parent = os.path.join(args.video_save_path, video_dir_root, exp, level)

            video_path_parent = os.path.join(args.video_root_path, video_dir_root, ref_angle, exp, level)
            video_clip_list = next(os.walk(video_path_parent))[1]
            for clip in video_clip_list:
                frame_save_dir = os.path.join(frame_save_parent, clip)
                frame_dir = os.path.join(video_path_parent, clip)
                frames = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))] # Read files , not folders
                frames.sort() # Sort the frames according to the frame number such that the frame in for loop can be extracted in order
                total_len_frames = len(frames)
                assert total_len_frames >= temporal_num+1, 'Frames are not enough to extract temporal ones!' # Check if the number of total frames is large than the number of temporal frames

                for frame in frames:
                    ref_frame_path = os.path.join(frame_dir, frame) # Reference image
                    ref_img = cv2.imread(ref_frame_path)

                    target_save_path = os.path.join(frame_save_dir, frame) # Use the ref frame number as the name of the folder name
                    ref_frame_num = int(frame[:-suffix_len]) # Convert the ref frame name to the frame number, i.e., remove the image suffix
                    for target in str_temporal_num_list: # Loop over the next few target frames
                        target_frame_number = ref_frame_num + int(target)*frame_interval # Compute the target frame number
                        target_frame_str = str(target_frame_number) + img_suffix # Obtaint the target frame name
                        target_frame_path = os.path.join(frame_dir, target_frame_str) # Obtain the target frame path
                        target_individual_save_path = os.path.join(target_save_path, target)
                        if not os.path.exists(target_individual_save_path): # Create a target individual folder if the folder does not exist
                            print('Creating target individual folders...')
                            os.makedirs(target_individual_save_path)
                        target_gt_img = cv2.imread(target_frame_path) 
                        
                        # Extract left target image
                        left_video_path_parent = os.path.join(args.video_root_path, video_dir_root, left_angle, exp, level) # Level folder
                        left_frame_dir = os.path.join(left_video_path_parent, clip) # Clip folder
                        left_target_frame_path = os.path.join(left_frame_dir, target_frame_str) # Target frame 
                        target_left_img = cv2.imread(left_target_frame_path) # Read by OpenCV

                        # Extract right target image
                        right_video_path_parent = os.path.join(args.video_root_path, video_dir_root. right_angle, exp, level)
                        right_frame_dir = os.path.join(right_video_path_parent, clip)
                        right_target_frame_path = os.path.join(right_frame_dir, target_frame_str)
                        target_right_img = cv2.imread(right_target_frame_path)

                        # Extract top target image
                        top_video_path_parent = os.path.join(args.video_root_path, video_dir_root, top_angle, exp, level)
                        top_frame_dir = os.path.join(top_video_path_parent, clip)
                        top_target_frame_path = os.path.join(top_frame_dir, target_frame_str)
                        target_top_img = cv2.imread(top_target_frame_path)

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix
    wfp_1 = f'examples/exp_data/{args.img_fp.split("/")[-1].replace(old_suffix, "")}'

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    depth_img = depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=False)
    P, pose = viz_pose_1(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)

    # Save RGB
    print('Data type of RGB image: ', img.dtype) # uint8
    # You do not have to convert BGR to RGB if you will use cv2.imwrite to save the image
    # However, you have to do the conversion if you want to use other way, say matplotlib, to visualize the imageb
    # img_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    rgb_save_name = wfp_1 + '_rgb' + '.jpg'
    cv2.imwrite(rgb_save_name, img)

    # Save 2D Landmarks
    print('Data type of 2D landmarks: ', ver_lst[0].dtype) # float32
    print('Shape of landmarks array: ', ver_lst[0].shape)
    landmarks2d_save_name = wfp_1 + '_2d' + '.npy'
    np.save(landmarks2d_save_name, ver_lst[0])

    # Save camera matrix
    print('Data type of cam matrix: ', P.dtype) # float32
    print('Shpae of camera matrix: ', P.shape)
    cam_matrix_save_name = wfp_1 + '_cam_mat' + '.npy'
    np.save(cam_matrix_save_name, P)

    # Save Pose (yaw; pitch, roll)
    pose_array = np.array(pose)
    print('Data type of pose: ', pose_array.dtype) # float64
    pose_save_name = wfp_1 + '_pose' + '.npy'
    np.save(pose_save_name, pose_array)

    # Save Depth
    print('Shape of depth image: ', depth_img.shape)
    print('Data type of depth: ', depth_img.dtype) # uint8
    depth_save_name = wfp_1 + '_depth' + '.npy'
    np.save(depth_save_name, depth_img)

    # if args.opt == '2d_sparse':
    #     draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    # elif args.opt == '2d_dense':
    #     draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    # elif args.opt == '3d':
    #     render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    # elif args.opt == 'depth':
    #     # if `with_bf_flag` is False, the background is black
    #     depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=False)
    # elif args.opt == 'pncc':
    #     pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    # elif args.opt == 'uv_tex':
    #     uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    # elif args.opt == 'pose':
    #     viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    # elif args.opt == 'ply':
    #     ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    # elif args.opt == 'obj':
    #     ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    # else:
    #     raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-r', '--video_root_path', type=str, default='/home/uss00067/Datasets/MEAD_Frames')
    parser.add_argument('-r', '--video_save_path', type=str, default='/home/uss00067/Datasets/FDC')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
