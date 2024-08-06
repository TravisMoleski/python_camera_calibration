import cv2
import numpy as np
import glob
# import yaml

import ruamel.yaml

def format2yaml(data):
    data = data.squeeze()
    # Convert NumPy array to byte string
    byte_string = data.tobytes()

    # Reconstruct the array from the byte string
    reconstructed_data = np.frombuffer(byte_string, dtype=data.dtype)
    reconstructed_data = reconstructed_data.reshape(data.shape)

    # Convert NumPy array to YAML
    return(reconstructed_data.tolist())


# Checkboard dimensions
CHECKERBOARD = (10,7)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-6)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

i_count = 0 
images = sorted(glob.glob('./images/*.png'))
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners2)
        i_count += 1
        
        # Draw and display the corners
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
 
cv2.destroyAllWindows()

# calculate K & D
N_imm = len(images)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]

retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

print(K)
print(D)
print(np.hstack((K, np.zeros((K.shape[0], 1), dtype=K.dtype))))
balance = 1
for fname in glob.glob("./images/*.png"):
    print(fname)
    img = cv2.imread(fname)
    img_dim = img.shape[:2][::-1]  

    scaled_K = K * img_dim[0] / img_dim[0]  
    scaled_K[2][2] = 1.0  
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
        img_dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
        new_K, img_dim, cv2.CV_16SC2)
    undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('img', undist_image)
    cv2.waitKey(1)

j_dict = {
    'camera_name': "usb_cam",
    'image_width':  img_dim[0],
    'image_height': img_dim[1],
    "distortion_model": 'equidistant',
    'camera_matrix': {
        "rows": K.shape[0],
        "cols": K.shape[1],
        "data": format2yaml(K)
    },
    'distortion_coefficients': {
        "rows": D.shape[0],
        "cols": D.shape[1],
        "data": format2yaml(D)
    },
    "rectification_matrix":{
    "rows": 3,
    "cols": 3,
    "data": format2yaml(np.eye(3))},
    "projection_matrix":{
        "rows": 3,
        "cols": 4,
        "data": format2yaml(np.hstack((K, np.zeros((K.shape[0], 1), dtype=K.dtype))))
    },
    "reprojectionError": None,
    "images_used": i_count

}
print(j_dict)

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=4, sequence=6, offset=3)
yaml.preserve_quotes = False
yaml.default_flow_style = None
yaml.default_style = None

with open('calibrate_kb.yaml', 'w') as outfile:
    yaml.dump(j_dict, outfile)
