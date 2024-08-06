import cv2
import os

# Open the video file
video_path = './video/1722958501.1269813.avi'  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a directory to save the frames
output_dir = './images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    # Save the frame as a PNG file
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {frame_count} frames and saved them to '{output_dir}' directory.")
