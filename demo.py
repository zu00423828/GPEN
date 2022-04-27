'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import pickle
import shutil
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm, trange
import __init_paths
from face_enhancement import FaceEnhancement
import subprocess
import face_alignment
from pathlib import Path
from mask import _cal_mouth_contour_mask
from bleed import LaplacianBlending


def brush_stroke_mask(img, color=(255, 255, 255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80

    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        if img is not None:
            mask = img  # Image.fromarray(img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(
                        2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)),
                          int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius,
                                     scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask


def extract_landmark(video_path, out_path):

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D)
    video = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count = int(video.get(7))
    for _ in trange(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        result = fa.get_landmarks(frame)
        if result is None:
            landmark = None
        else:
            landmark = result[0]
        all_landmarks.append(landmark)

    with open(out_path, 'wb') as f:
        pickle.dump(all_landmarks, f)
    return out_path


def process(vidoe_path, pkl, out_path):
    processer = FaceEnhancement(in_size=args.in_size, model=args.model, use_sr=args.use_sr, sr_model=args.sr_model, sr_scale=args.sr_scale,
                                channel_multiplier=args.channel_multiplier, narrow=args.narrow, key=args.key, device='cuda')
    full_video = cv2.VideoCapture(vidoe_path)
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    fps = full_video.get(5)
    frame_count = int(full_video.get(7))
    temp_path = 'temp.avi'
    out_video = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    # f = open(pkl, 'rb')
    # isface_list = pickle.load(f)
    chunk_video = cv2.VideoCapture('chunk.avi')
    chunk_counts = int(chunk_video.get(7))
    for i in trange(chunk_counts):
        _, frame = chunk_video.read()
        out_video.write(frame)
    full_video.set(1, chunk_counts)
    for i in trange(chunk_counts, frame_count):
        _, frame = full_video.read()
        # frame = cv2.resize(frame, (w//2, h//2))
        img_out, _, _ = processer.process(
            frame, aligned=False)
        img_out = cv2.resize(img_out, (w, h))
        out_video.write(img_out)

    shutil.copyfile(temp_path, 'video/e'+Path(vidoe_path).stem+'.avi')
    command = f"ffmpeg -y -i {vidoe_path} temp.wav "
    subprocess.call(command, shell=True)
    command = f"ffmpeg -y -i {temp_path} -i temp.wav -vcodec h264 {out_path} "
    subprocess.call(command, shell=True)


def create_lb(iters: int = None, ksize: int = 3, sigma=0, device='cpu'):
    lb = LaplacianBlending(
        sigma=sigma,
        ksize=ksize,
        iters=4 if iters is None else iters).to(device).eval()
    for param in lb.parameters():
        param.requires_grad = False
    return lb


def quantize_position(x1, x2, y1, y2, iters=None):
    w = x2 - x1
    h = y2 - y1
    x_center = (x2 + x1) // 2
    y_center = (y2 + y1) // 2
    half_w = np.math.ceil(w / (2 ** iters)) * \
        (2 ** (iters - 1))
    half_h = np.math.ceil(h / (2 ** iters)) * \
        (2 ** (iters - 1))
    x1 = x_center - half_w
    x2 = x_center + half_w
    y1 = y_center - half_h
    y2 = y_center + half_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    return int(x1), int(x2), int(y1), int(y2)


def blur_video_mouth(video_path, pkl, out_path, device='cpu'):
    f = open(pkl, 'rb')
    landmarks = pickle.load(f)
    video = cv2.VideoCapture(video_path)
    h = int(video.get(4))
    w = int(video.get(3))
    fps = video.get(5)
    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    lb = create_lb(4)
    for i in trange(len(landmarks)):
        _, frame = video.read()
        x1, x2, y1, y2 = quantize_position(0, w, 0, h, 4)
        mask = _cal_mouth_contour_mask(landmarks[i], y2, x2, None, 0.1)
        l_min_x = int(np.min(landmarks[0][:, 0]))-80
        l_max_x = int(np.max(landmarks[0][:, 0]))+80
        l_min_y = int(np.min(landmarks[0][:, 1]))-80
        l_max_y = int(np.max(landmarks[0][:, 1]))+80
        frame = cv2.copyMakeBorder(frame, 0, max(
            0, y2-frame.shape[0]), 0, max(0, x2-frame.shape[1]), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        mouth = np.zeros((y2, x2, 3))
        mouth[l_min_y:l_max_y,
              l_min_x:l_max_x] = frame[l_min_y:l_max_y, l_min_x:l_max_x]
        blur_mouth = cv2.GaussianBlur(mouth, (7, 7), 0)
        blur_mouth = cv2.copyMakeBorder(blur_mouth.astype(np.uint8), 0, max(
            0, y2-blur_mouth.shape[0]), 0, max(0, x2-blur_mouth.shape[1]), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        mask_t = torch.tensor(
            mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        moth_t = torch.tensor(blur_mouth/255, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0).to(device)
        origin_t = torch.tensor(frame/255, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0).to(device)
        # print(origin_t.shape, moth_t.shape, mask_t.shape)
        out = lb(origin_t, moth_t, mask_t)
        full_out = (out[0][:, :h, :w].permute(
            1, 2, 0)*255).numpy().astype(np.uint8)
        out_video.write(full_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--task', type=str,
                        default='FaceEnhancement', help='task of GPEN model')
    parser.add_argument('--key', type=str, default=None,
                        help='key of GPEN model')
    parser.add_argument('--in_size', type=int, default=512,
                        help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None,
                        help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int,
                        default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1,
                        help='channel narrow scale')
    parser.add_argument('--use_sr', action='store_true',
                        help='use sr or not')
    parser.add_argument('--use_cuda', action='store_true',
                        help='use cuda or not')
    parser.add_argument('--save_face', action='store_true',
                        help='save face or not')
    parser.add_argument('--aligned', action='store_true',
                        help='input are aligned faces or not')
    parser.add_argument('--sr_model', type=str,
                        default='realesrnet', help='SR model')
    parser.add_argument('--sr_scale', type=int,
                        default=2, help='SR scale')
    parser.add_argument('--indir', type=str,
                        default='examples/imgs', help='input folder')
    parser.add_argument('--outdir', type=str,
                        default='results/outs-BFR', help='output folder')
    args = parser.parse_args()

    # extract_landmark('video/1.mp4', '1.pkl')
    # process('video/2.mp4', '2.pkl', 'video/enhance2.mp4')
    # process('video/3.mp4','video/enhance3.mp4')

    # extract_landmark('video/5.mp4', 'video/pkl/5.pkl')
    # extract_landmark('video/6.mp4', 'video/pkl/6.pkl')
    # process('blur.avi', 'video/pkl/4.pkl', 'video/out/enhance4_new.mp4')

    # extract_landmark('video/4.mp4', 'video/pkl/4.pkl')
    # blur_video_mouth('video/4.mp4', 'video/pkl/4.pkl', 'blur.avi')
    # process('video/5.mp4', 'video/pkl/5.pkl', 'video/out/enhance5.mp4')
    process('video/6.mp4', 'video/pkl/6.pkl', 'video/out/enhance6.mp4')
