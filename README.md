# Chessboard AR House: Camera Pose Estimation and AR

## Description
This project performs camera pose estimation and augmented reality (AR) visualization using OpenCV and a printed chessboard pattern.

The camera pose is estimated from detected chessboard corners using the calibration result obtained in Homework #3.  
After estimating the camera pose, a custom 3D AR object is projected onto the chessboard.

This project uses a custom **AR House** object, which is different from the simple example object provided in the lecture.

---

## Features
- Detects chessboard corners from input video
- Estimates camera pose using `cv.solvePnP()`
- Uses camera calibration parameters from Homework #3
- Projects a custom 3D AR object onto the chessboard
- Displays camera pose information on the image
- Saves the AR result as a demo video

---

## Environment
- Python 3.x
- OpenCV
- NumPy

---

## Chessboard Pattern
- Paper size: A4
- Square size: **25 mm (0.025 m)**
- Pattern: **10 × 7 inner corners (vertices)**
- Total squares: **11 × 8**

---

## Files
- `pose_estimation_ar.py` — camera pose estimation and AR visualization
- `calibration_result.npz` — camera calibration result from Homework #3
- `ar_demo.mp4` — AR demo video
- `README.md` — project description and demo

---

## How to Run

### 1. Prepare files
Place the following files in the same folder:
- `pose_estimation_ar.py`
- `calibration_result.npz`
- `chessboard.mov`

### 2. Run the program
```bash
python pose_estimation_ar.py --calib calibration_result.npz --video chessboard.mov --board_w 10 --board_h 7 --cellsize 0.025
controls
- ESC — quit
- I — toggle pose information
```

## Camera Calibration Result Used

The following camera calibration result from Homework #3 was used:
- Number of applied images: 20
- RMS error (OpenCV return): 1.298914
- Reprojection RMSE: 1.298914

## Camera Matrix (K)
[[1.07339050e+03 0.00000000e+00 6.35329353e+02]
 [0.00000000e+00 1.00426491e+03 3.31775263e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

## Distortion Coefficients
[ 0.21716506 -0.96385693 -0.02145939 -0.00209672  1.41297032]

## AR Object

# The AR object used in this project is a custom 3D House model.
The model includes:
- base
- walls
- roof
- door
- window
- chimney
This object was intentionally designed to be different from the lecture example.

## Demo
# Screenshot
<img width="1919" height="1079" alt="ar_demo" src="https://github.com/user-attachments/assets/e38f81ea-3f57-4378-8737-9759a4bd83ab" />

# Video
![Video demo](ar_demo.gif)

## Example Output

The program successfully:
- Detected chessboard corners
- Estimated the camera pose
- Projected the AR house onto the chessboard
- Generated an AR visualization video

## Notes
- The chessboard should be clearly visible in the video
- The calibration result must be obtained from the same camera used for the AR video
- If chessboard detection fails, AR visualization will not be displayed
- A better calibration result can improve AR stability and accuracy

## Conclusion

This project successfully fulfills both requirements of Homework #4:
- Camera pose estimation
- AR object visualization
Using the calibration result from Homework #3, the program estimates the camera pose from a chessboard pattern and renders a custom AR house on the board.
