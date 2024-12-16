import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# Example homography matrix (replace with your actual matrix)
H = np.array([ 
    [1.12743519e+00, -1.77414144e-01, -9.17122803e+02],
 [ 2.31532982e-01,  1.08190601e+00,  5.60285645e+02],
 [ 3.61788091e-05, -8.30680372e-06,  1.00000000e+00]
 ])

# Normalize the homography matrix
H = H / H[2, 2]

# Camera intrinsic matrix (example, adjust to your camera)
K = np.array([
    [1000, 0, 320],
    [0, 1000, 240],
    [0, 0, 1]
])

# Decompose the homography matrix
retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

# Extract the first rotation matrix (if multiple solutions exist)
R_matrix = rotations[0]  # Choose the correct solution based on your setup

# Convert the rotation matrix to Euler angles
rotation_obj = R.from_matrix(R_matrix)
euler_angles = rotation_obj.as_euler('xyz', degrees=True)  # 'xyz' for Tait-Bryan angles

# Output the results
print("Rotation Matrix:")
print(R_matrix)
print("\nEuler Angles (degrees):")
print(f"Yaw (Z): {euler_angles[2]:.2f}, Pitch (Y): {euler_angles[1]:.2f}, Roll (X): {euler_angles[0]:.2f}")



H = np.array([ 
[8.63743891e-01,  1.48629140e-01,  7.05418331e+02],
 [-1.67450199e-01,  9.22344657e-01, -6.71623816e+02],
 [-3.23331446e-05,  2.96467188e-06,  1.00000000e+00]
 ])


# Normalize the homography matrix
H = H / H[2, 2]

# Camera intrinsic matrix (example, adjust to your camera)
K = np.array([
    [1000, 0, 320],
    [0, 1000, 240],
    [0, 0, 1]
])

# Decompose the homography matrix
retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

# Extract the first rotation matrix (if multiple solutions exist)
R_matrix = rotations[0]  # Choose the correct solution based on your setup

# Convert the rotation matrix to Euler angles
rotation_obj = R.from_matrix(R_matrix)
euler_angles = rotation_obj.as_euler('xyz', degrees=True)  # 'xyz' for Tait-Bryan angles

# Output the results
print("Rotation Matrix:")
print(R_matrix)
print("\nEuler Angles (degrees):")
print(f"Yaw (Z): {euler_angles[2]:.2f}, Pitch (Y): {euler_angles[1]:.2f}, Roll (X): {euler_angles[0]:.2f}")\


H_inv = np.linalg.inv(R_matrix)
print("Inv Matrix:")
print(H_inv)
