import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import random
from ruamel.yaml import YAML
from concurrent.futures import ThreadPoolExecutor

CHECKERBOARD = (10, 7)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-6)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                    cv2.fisheye.CALIB_CHECK_COND + \
                    cv2.fisheye.CALIB_FIX_SKEW

images_all = sorted(glob.glob('./images/*.png'))
objp_template = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp_template[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

def format2yaml(data):
    return data.squeeze().flatten().tolist()

def compute_reprojection_error(obj_points, img_points, K, D, R, T):
    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], R[i], T[i], K, D)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    return total_error / len(obj_points)

def process_image(fname):
    try:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            refined = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            return fname, objp_template.copy(), refined, gray.shape[::-1]
    except:
        pass
    return fname, None, None, None

yaml = YAML()
yaml.indent(mapping=4, sequence=6, offset=3)
yaml.preserve_quotes = False
yaml.default_flow_style = None

print("PRUNING IMAGES TO REMOVE NON CHECKERS...")
    # Step 1: Parallel corner detection
objpoints, imgpoints, img_shape = [], [], None
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, images_all))
for fname, obj, imgpt, shape in results:
    if obj is not None:
        objpoints.append(obj)
        imgpoints.append(imgpt)
        img_shape = shape  # use the first valid image for dimensions
    else:
        try:
            os.remove(fname)
        except FileNotFoundError:
            pass

images_all = sorted(glob.glob('./images/*.png'))
print("EXTRACT CALIBRATION IMGS")
for i_count_max in tqdm(range(1, len(images_all))):

    images = random.sample(images_all, i_count_max)

    # Step 1: Parallel corner detection
    objpoints, imgpoints, img_shape = [], [], None
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, images))

    for fname, obj, imgpt, shape in results:
        if obj is not None:
            objpoints.append(obj)
            imgpoints.append(imgpt)
            img_shape = shape  # use the first valid image for dimensions
        else:
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

    if len(objpoints) < 3:
        continue  # Skip if not enough points for calibration

    # Step 2: Calibrate
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]

    try:
        _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape, K, D, rvecs, tvecs,
            calibration_flags, subpix_criteria)
    except:
        continue

    reprojection_error = compute_reprojection_error(objpoints, imgpoints, K, D, rvecs, tvecs)

    j_dict = {
        'camera_name': "usb_cam",
        'image_width':  img_shape[0],
        'image_height': img_shape[1],
        "distortion_model": 'equidistant',
        'camera_matrix': {
            "rows": K.shape[0], "cols": K.shape[1], "data": format2yaml(K)},
        'distortion_coefficients': {
            "rows": D.shape[0], "cols": D.shape[1], "data": format2yaml(D)},
        "rectification_matrix": {
            "rows": 3, "cols": 3, "data": format2yaml(np.eye(3))},
        "projection_matrix": {
            "rows": 3, "cols": 4, "data": format2yaml(np.hstack((K, np.zeros((K.shape[0], 1))))),
        },
        "reprojectionError": reprojection_error,
        "images_used": len(objpoints)
    }

    os.makedirs('./caliRandom', exist_ok=True)
    with open(f'./caliRandom/{i_count_max}.yaml', 'w') as outfile:
        yaml.dump(j_dict, outfile)
