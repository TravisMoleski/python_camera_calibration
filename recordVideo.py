import cv2
import time

# Open a connection to the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 24)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the codec and create VideoWriter object
# 'XVID' is the codec, 'output.avi' is the filename, 20.0 is the frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
name = './video/'+str(time.time()) + '.avi'
out = cv2.VideoWriter(name, fourcc, 24.0, (int(cap.get(3)), int(cap.get(4))))

print("Recording... Press 'q' to stop.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording finished.")
