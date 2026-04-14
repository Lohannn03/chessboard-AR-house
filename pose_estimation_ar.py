import cv2 as cv
import numpy as np
import argparse


def parse_video_source(src: str):
    return int(src) if src.isdigit() else src


def load_calibration(npz_file: str):
    data = np.load(npz_file)
    K = data["K"]
    dist_coeff = data["dist_coeff"]
    return K, dist_coeff


def build_chessboard_points(board_pattern, board_cellsize):
    return np.array(
        [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])],
        dtype=np.float32
    ) * board_cellsize


def build_house_model(board_cellsize):
    s = board_cellsize

    base = np.array([
        [2.0, 2.0, 0.0],
        [6.0, 2.0, 0.0],
        [6.0, 5.0, 0.0],
        [2.0, 5.0, 0.0],
    ], dtype=np.float32) * s

    wall = np.array([
        [2.0, 2.0, -2.0],
        [6.0, 2.0, -2.0],
        [6.0, 5.0, -2.0],
        [2.0, 5.0, -2.0],
    ], dtype=np.float32) * s

    ridge = np.array([
        [4.0, 2.0, -3.2],
        [4.0, 5.0, -3.2],
    ], dtype=np.float32) * s

    door = np.array([
        [3.2, 2.0, 0.0],
        [4.0, 2.0, 0.0],
        [4.0, 2.0, -1.2],
        [3.2, 2.0, -1.2],
    ], dtype=np.float32) * s

    window = np.array([
        [6.0, 3.0, -0.8],
        [6.0, 4.0, -0.8],
        [6.0, 4.0, -1.5],
        [6.0, 3.0, -1.5],
    ], dtype=np.float32) * s

    chimney_bottom = np.array([
        [4.6, 3.4, -2.0],
        [5.1, 3.4, -2.0],
        [5.1, 4.0, -2.0],
        [4.6, 4.0, -2.0],
    ], dtype=np.float32) * s

    chimney_top = np.array([
        [4.6, 3.4, -3.1],
        [5.1, 3.4, -3.1],
        [5.1, 4.0, -3.1],
        [4.6, 4.0, -3.1],
    ], dtype=np.float32) * s

    return {
        "base": base,
        "wall": wall,
        "ridge": ridge,
        "door": door,
        "window": window,
        "chimney_bottom": chimney_bottom,
        "chimney_top": chimney_top,
    }


def draw_polyline(img, pts2d, closed, color, thickness=2):
    cv.polylines(img, [np.int32(pts2d)], closed, color, thickness, cv.LINE_AA)


def draw_house(img, rvec, tvec, K, dist_coeff, house):
    proj = {}
    for key, pts3d in house.items():
        pts2d, _ = cv.projectPoints(pts3d, rvec, tvec, K, dist_coeff)
        proj[key] = pts2d.reshape(-1, 2)

    base = proj["base"]
    wall = proj["wall"]
    ridge = proj["ridge"]
    door = proj["door"]
    window = proj["window"]
    cb = proj["chimney_bottom"]
    ct = proj["chimney_top"]

    draw_polyline(img, base, True, (80, 180, 255), 2)
    draw_polyline(img, wall, True, (255, 120, 80), 2)

    for b, w in zip(base, wall):
        cv.line(img, np.int32(b), np.int32(w), (0, 255, 0), 2, cv.LINE_AA)

    cv.line(img, np.int32(wall[0]), np.int32(ridge[0]), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(img, np.int32(wall[1]), np.int32(ridge[0]), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(img, np.int32(wall[3]), np.int32(ridge[1]), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(img, np.int32(wall[2]), np.int32(ridge[1]), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(img, np.int32(ridge[0]), np.int32(ridge[1]), (0, 0, 255), 2, cv.LINE_AA)

    draw_polyline(img, door, True, (255, 255, 0), 2)
    draw_polyline(img, window, True, (255, 0, 255), 2)

    draw_polyline(img, cb, True, (180, 180, 180), 2)
    draw_polyline(img, ct, True, (180, 180, 180), 2)
    for p1, p2 in zip(cb, ct):
        cv.line(img, np.int32(p1), np.int32(p2), (180, 180, 180), 2, cv.LINE_AA)

    label_anchor = np.int32(ridge[0] + np.array([10, -10]))
    cv.putText(img, 'AR House', tuple(label_anchor), cv.FONT_HERSHEY_DUPLEX,
               0.6, (0, 255, 255), 2, cv.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Camera Pose Estimation and AR')
    parser.add_argument('--calib', default='calibration_result.npz', help='calibration result from HW3')
    parser.add_argument('--video', default='chessboard.mov', help='webcam index like 0, or video file path')
    parser.add_argument('--board_w', type=int, default=10, help='number of inner corners along width')
    parser.add_argument('--board_h', type=int, default=7, help='number of inner corners along height')
    parser.add_argument('--cellsize', type=float, default=0.025, help='square size in meters')
    parser.add_argument('--save', default='ar_demo.mp4', help='output demo video filename')
    args = parser.parse_args()

    board_pattern = (args.board_w, args.board_h)
    obj_points = build_chessboard_points(board_pattern, args.cellsize)
    house = build_house_model(args.cellsize)

    K, dist_coeff = load_calibration(args.calib)

    cap = cv.VideoCapture(parse_video_source(args.video))
    assert cap.isOpened(), f'Cannot open video source: {args.video}'

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    writer = None
    show_info = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, img_points = cv.findChessboardCorners(gray, board_pattern)

        view = frame.copy()

        if found:
            img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(view, board_pattern, img_points, found)

            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
            if ret:
                draw_house(view, rvec, tvec, K, dist_coeff, house)

                R, _ = cv.Rodrigues(rvec)
                cam_pos = (-R.T @ tvec).flatten()

                if show_info:
                    info1 = f'Camera XYZ: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}] m'
                    info2 = f'rvec: [{rvec[0,0]:.3f}, {rvec[1,0]:.3f}, {rvec[2,0]:.3f}]'
                    cv.putText(view, info1, (10, 25), cv.FONT_HERSHEY_DUPLEX,
                               0.55, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(view, info2, (10, 50), cv.FONT_HERSHEY_DUPLEX,
                               0.50, (0, 255, 255), 1, cv.LINE_AA)
        else:
            cv.putText(view, 'Chessboard not detected', (10, 25), cv.FONT_HERSHEY_DUPLEX,
                       0.7, (0, 0, 255), 2, cv.LINE_AA)

        cv.putText(view, 'ESC: quit | I: toggle info', (10, view.shape[0] - 15),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        if writer is None:
            h, w = view.shape[:2]
            fps = cap.get(cv.CAP_PROP_FPS)
            if fps <= 1e-6:
                fps = 30.0
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            writer = cv.VideoWriter(args.save, fourcc, fps, (w, h))

        writer.write(view)
        cv.imshow('Camera Pose Estimation and AR', view)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('i'), ord('I')):
            show_info = not show_info

    cap.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
