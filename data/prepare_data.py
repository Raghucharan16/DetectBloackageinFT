# For CPU
import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_video(input_path, output_dir, img_size=160):
    """Convert videos to normalized numpy arrays"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    
    # Save as memory-mapped file
    base_name = os.path.basename(input_path).split('.')[0]
    np.save(os.path.join(output_dir, f"{base_name}.npy"), np.array(frames))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    
    for video in tqdm(os.listdir(args.input_dir)):
        preprocess_video(os.path.join(args.input_dir, video), args.output_dir)
