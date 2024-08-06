import cv2
import numpy as np
import glob
# import yaml
import os
import random
import ruamel.yaml

def format2yaml(data):
    data = data.squeeze()
    # Convert NumPy array to byte string
    byte_string = data.tobytes()

    # Reconstruct the array from the byte string
    reconstructed_data = np.frombuffer(byte_string, dtype=data.dtype)
    reconstructed_data = reconstructed_data.reshape(data.shape)

    # Convert NumPy array to YAML
    reconstructed_data = reconstructed_data.flatten()
    return(reconstructed_data.tolist())


def compute_reprojection_error(obj_points, imgpoints2, K, D, R, T):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], R[i], T[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    return mean_error/len(objpoints)


# Checkboard dimensions
CHECKERBOARD = (10,7)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-6)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
images = sorted(glob.glob('./images/*.png'))
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

# objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
maxList = np.arange(1, len(images), 1)
# maxList = np.arange(1, 30, 1)

for i_count_max in maxList:
    images = sorted(glob.glob('./images/*.png'))
    images = random.sample(images, i_count_max)

    i_count_max = int(i_count_max)
    print(i_count_max)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    i_count = 0 
    i = 0
    # images = sorted(glob.glob('./images/*.png'))
    # for fname in images:
    while i_count < i_count_max:
        fname = images[i]
        print(fname)
        try:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            continue

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners2)
            i_count += 1

            print(i_count, i_count_max)
            
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(1)
        else:
            os.remove(fname) 

        i += 1
    
    cv2.destroyAllWindows()

    # calculate K & D
    N_imm = len(images)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]

    try:
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
    except:
        continue

    print(K)
    print(D)
    print(np.hstack((K, np.zeros((K.shape[0], 1), dtype=K.dtype))))
    reprojection_error = compute_reprojection_error(objpoints, imgpoints, K, D, rvecs, tvecs)
    print(f"Reprojection Error: {reprojection_error}")
    img_dim = img.shape[:2][::-1]  

    balance = 1
    # for fname in glob.glob("./images/*.png"):

    # print(fname)
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

    # cv2.imshow('img', undist_image)
    # cv2.waitKey(1000)

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
        "reprojectionError": reprojection_error,
        "images_used": i_count

    }
    print(j_dict)

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.preserve_quotes = False
    yaml.default_flow_style = None
    yaml.default_style = None

    dumpStr = './caliRandom/'+str(i_count_max)+'.yaml'
    with open(dumpStr, 'w') as outfile:
        yaml.dump(j_dict, outfile)
