import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

video_path = './video/1753451742.9740436.avi'
output_dir = './images'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
frames = []

# Read all frames into memory (fast with disk-based video)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append((frame_count, frame.copy()))
    frame_count += 1

cap.release()

# Write function
def save_frame(args):
    idx, frame = args
    filename = os.path.join(output_dir, f'frame_{idx:04d}.png')
    cv2.imwrite(filename, frame)

# Use ThreadPoolExecutor to write frames in parallel
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(save_frame, frames), total=len(frames), desc="Saving frames"))

print(f"Extracted and saved {frame_count} frames to '{output_dir}'.")
