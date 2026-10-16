"""
calibration.py - overhead camera -> robot arm base frame, via the camera's 3D pose.

Phase 3a of the Vision-Guided Robotic Sorting Arm. Replaces table_calibration.py
(homography) with the full 3D version of the same idea: instead of hand-measuring
where the arm is relative to the camera (ARM_POSITION), the camera's pose in the
arm frame is SOLVED from the fixed ArUco markers at measured table positions.

Commands:
    python calibration.py calibrate   # solve + save the camera pose in the arm frame
    python calibration.py markers     # live: every marker's position in the ARM frame
    python calibration.py validate    # live: arm-frame grid + reach band; click for coords
    python calibration.py check       # has the camera moved since calibration?

Runtime use in the pick pipeline:
    from camera_to_arm import CameraToArm
    cam = CameraToArm()
    x, y, z = cam.pixel_to_arm(u, v, z_cm=1.25)   # z_cm = height of the point you're locating
    reach, pan_deg = cam.to_polar(x, y)            # -> IK solver

Arm frame (centimetres - same units as the IK solver):
    origin : on the table, directly below the shoulder_pan rotation axis
    +x     : straight ahead of the arm (pan = 0)
    +y     : to the arm's left
    +z     : up. The table surface is z = 0.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

CAMERA_INDEX = 0                         
FRAME_W, FRAME_H = 640, 480              
INTRINSICS_FILE = Path("calibration/camera_calibration.yaml")
POSE_FILE = Path("calibration/camera_pose.json")
ARUCO_DICT = cv.aruco.DICT_4X4_50

# Fixed calibration markers: CENTRE of each marker in the arm frame (cm), on the table.
# Placeholders - tape them down, then measure from the point below the pan axis.
# Only centres are used, so marker size and rotation on the table don't matter.
# Use 5-6 markers spread around the workspace edge, outside the arm's sweep.
MARKER_POSITIONS_CM = {
    0: (7.5, -13.5),
    1: (19.0, -9.0),
    2: (25.5, -5.5),
    3: (26.5, 4.0),
    4: (24.0, 12.0),
    5: (26.0, -14.0)
}

REACH_BAND_CM = (9.0, 27.0)     # usable band from the IK validation
SHOULDER_HEIGHT_CM = 12.0       # drawn in `validate` as a sanity check on the z axis
REPROJ_WARN_PX = 5.0            # calibration fit quality threshold
DRIFT_WARN_PX = 8.0             # marker shift that means "camera moved, recalibrate"


# --------------------------------------------------------------------------
# Pure helpers (no camera needed - these are what the tests exercise)
# --------------------------------------------------------------------------
def load_intrinsics(path=INTRINSICS_FILE):
    """(K, dist) from the checkerboard YAML. Required for the 3D method."""
    global FRAME_W, FRAME_H
    if not Path(path).exists():
        sys.exit(f"{path} not found - the pose method needs the checkerboard calibration")
    data = yaml.safe_load(Path(path).read_text())
    K = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(1, -1)
    FRAME_W = int(data.get("image_width", FRAME_W))
    FRAME_H = int(data.get("image_height", FRAME_H))
    return K, dist


def marker_centre_px(corners, K, dist):
    """Undistorted pixel position of a marker's true centre.

    Uses the intersection of the diagonals rather than the mean of the four
    corners: under perspective the corner mean is NOT the projected centre.
    Output is in undistorted pixel coordinates (same K, zero distortion).
    """
    pts = cv.undistortPoints(np.asarray(corners, np.float64).reshape(-1, 1, 2),
                             K, dist, P=K).reshape(4, 2)
    h = np.hstack([pts, np.ones((4, 1))])
    p = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
    return p[:2] / p[2]


def solve_camera_pose(centres_px, marker_positions_cm, K):
    """Camera pose from marker centres (undistorted px) at known arm-frame positions.

    Returns (rvec, tvec, ids, reproj_err_px). rvec/tvec map ARM-frame points
    into CAMERA-frame points: p_cam = R @ p_arm + tvec. tvec is therefore the
    arm origin expressed in the camera frame - the thing ARM_POSITION was
    trying to measure by hand.
    """
    ids = [i for i in marker_positions_cm if i in centres_px]
    if len(ids) < 4:
        raise ValueError(f"need >= 4 calibration markers, found {len(ids)}")
    obj = np.array([[*marker_positions_cm[i], 0.0] for i in ids], dtype=np.float64)
    img = np.array([centres_px[i] for i in ids], dtype=np.float64)
    ok, rvec, tvec = cv.solvePnP(obj, img, K, None, flags=cv.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed - are the marker positions collinear?")
    rvec, tvec = cv.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    proj = cv.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
    return rvec, tvec, ids, np.linalg.norm(proj - img, axis=1)


# --------------------------------------------------------------------------
# Runtime mapper
# --------------------------------------------------------------------------
class CameraToArm:
    """Image pixels <-> arm base frame, using the solved camera pose."""

    def __init__(self, pose_path=POSE_FILE, K=None, dist=None, rvec=None, tvec=None):
        if rvec is None:
            if not Path(pose_path).exists():
                raise FileNotFoundError(f"{pose_path} not found - run `calibrate` first")
            cfg = json.loads(Path(pose_path).read_text())
            rvec, tvec = cfg["rvec"], cfg["tvec"]
            self.marker_px = {int(k): np.array(v) for k, v in cfg["marker_px"].items()}
        else:
            self.marker_px = {}
        if K is None:
            K, dist = load_intrinsics()
        self.K, self.dist = K, dist
        self.K_inv = np.linalg.inv(K)
        self.rvec = np.asarray(rvec, np.float64).reshape(3, 1)
        self.tvec = np.asarray(tvec, np.float64).reshape(3, 1)
        self.R, _ = cv.Rodrigues(self.rvec)
        # camera optical centre in the arm frame
        self.C = (-self.R.T @ self.tvec).ravel()

    def pixel_to_arm(self, u, v, z_cm=0.0, undistorted=False):
        """Back-project an image point onto the horizontal plane z = z_cm.

        z_cm is the height of the point you're locating: 0 for something on
        the table, the full height for a top-face centre, roughly half the
        height for a segmentation-mask centroid. Parallax is handled exactly
        by the geometry - no correction factor needed.
        """
        if undistorted:
            n = self.K_inv @ np.array([u, v, 1.0])
        else:
            n = cv.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.dist).reshape(2)
            n = np.array([n[0], n[1], 1.0])
        d = self.R.T @ n                         # ray direction in the arm frame
        if abs(d[2]) < 1e-9:
            raise ValueError("ray is parallel to the table")
        s = (z_cm - self.C[2]) / d[2]
        p = self.C + s * d
        return float(p[0]), float(p[1]), float(p[2])

    def camera_point_to_arm(self, p_cam):
        """A 3D point in the CAMERA frame (e.g. a marker tvec, in cm) -> arm frame.

        This is the correct version of `ARM_POSITION - tvec`: it removes the
        camera->arm ROTATION as well as the offset.
        """
        p_cam = np.asarray(p_cam, np.float64).reshape(3, 1)
        return (self.R.T @ (p_cam - self.tvec)).ravel()

    def arm_to_pixel(self, points_cm):
        """Arm-frame 3D points -> raw (distorted) image pixels, for overlays."""
        pts = np.asarray(points_cm, np.float64).reshape(-1, 1, 3)
        return cv.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)[0].reshape(-1, 2)

    @staticmethod
    def to_polar(x, y):
        """(x, y) cm -> (horizontal reach cm, shoulder_pan degrees).

        Reach is from the pan axis. If the IK measures reach from the
        shoulder_lift axis and that sits forward of the pan axis, subtract
        that offset before calling the solver.
        """
        return math.hypot(x, y), math.degrees(math.atan2(y, x))

    def camera_summary(self):
        """Camera position (arm frame, cm) and tilt of the optical axis from vertical."""
        optical_axis = self.R.T @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(np.clip(-optical_axis[2], -1, 1)))
        return self.C, tilt

    def drift_px(self, centres):
        shifts = [np.linalg.norm(centres[i] - p) for i, p in self.marker_px.items() if i in centres]
        return max(shifts) if shifts else float("inf")


# --------------------------------------------------------------------------
# Camera helpers
# --------------------------------------------------------------------------
def make_detector():
    return cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(ARUCO_DICT),
                                  cv.aruco.DetectorParameters())


def open_camera():
    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    got = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    if got != (FRAME_W, FRAME_H):
        cap.release()
        sys.exit(f"Camera gave {got[0]}x{got[1]} but calibration is for {FRAME_W}x{FRAME_H}")
    return cap


def detect_corners(cap, detector, n_frames=20):
    """Average each marker's corners over n_frames. Returns ({id: (4,2)}, last_frame)."""
    acc, frame = {}, None
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            sys.exit("Failed to read frame from camera")
        corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            acc.setdefault(int(i), []).append(c.reshape(4, 2))
    return {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= n_frames // 2}, frame


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_calibrate(_):
    K, dist = load_intrinsics()
    cap = open_camera()
    try:
        corners, frame = detect_corners(cap, make_detector())
    finally:
        cap.release()

    centres = {i: marker_centre_px(c, K, dist) for i, c in corners.items()}
    missing = sorted(set(MARKER_POSITIONS_CM) - set(centres))
    if missing:
        print(f"WARNING: calibration markers not detected: {missing}")
    try:
        rvec, tvec, ids, err_px = solve_camera_pose(centres, MARKER_POSITIONS_CM, K)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))
    if len(ids) == 4:
        print("NOTE: only 4 markers - use 5-6 so the residuals actually test the fit")

    cam = CameraToArm(K=K, dist=dist, rvec=rvec, tvec=tvec)
    print("\nPer-marker fit:            reproj px   table error mm")
    for i, e in zip(ids, err_px):
        x, y, _ = cam.pixel_to_arm(*centres[i], undistorted=True)
        mx, my = MARKER_POSITIONS_CM[i]
        print(f"  id {i}:                      {e:5.2f}        {math.hypot(x - mx, y - my) * 10:5.2f}")
    if err_px.max() > REPROJ_WARN_PX:
        print(f"  -> reprojection > {REPROJ_WARN_PX}px: re-measure marker positions, check the "
              "zoom/focus haven't changed since the checkerboard calibration")

    C, tilt = cam.camera_summary()
    print(f"\nCamera position in arm frame: x={C[0]:.1f}  y={C[1]:.1f}  height={C[2]:.1f} cm")
    print(f"Camera tilt from vertical:    {tilt:.1f} deg")
    print("  -> check the height against a tape measure; it should agree within ~1 cm")
    t = tvec.ravel()
    print(f"Arm origin in camera frame:   ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) cm   "
          "(what ARM_POSITION was estimating by hand)")

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    POSE_FILE.write_text(json.dumps({
        "rvec": rvec.ravel().tolist(),
        "tvec": tvec.ravel().tolist(),
        "frame_size": [FRAME_W, FRAME_H],
        "camera_position_cm": C.tolist(),
        "tilt_deg": tilt,
        "marker_px": {str(i): centres[i].tolist() for i in ids},
        "reproj_err_px": {str(i): float(e) for i, e in zip(ids, err_px)},
    }, indent=2))

    vis = frame.copy()
    for i in ids:
        u, v = cam.arm_to_pixel([[*MARKER_POSITIONS_CM[i], 0]])[0].astype(int)
        cv.drawMarker(vis, (u, v), (0, 255, 0), cv.MARKER_CROSS, 14, 2)
        cv.putText(vis, str(i), (u + 8, v - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imwrite(str(POSE_FILE.parent / "calibration_frame.png"), vis)
    print(f"\nSaved {POSE_FILE}")


def cmd_markers(_):
    """Your pose script, finished: every marker's position in the ARM frame.

    Calibration markers show their error against the measured position.
    Any other marker (e.g. a loose one you move around) shows where the
    arm should go to reach it - check it with a ruler.
    """
    cam = CameraToArm()
    cap = open_camera()
    detector = make_detector()
    font = cv.FONT_HERSHEY_SIMPLEX
    print("p: print positions   q: quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            corners, ids, _ = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            lines = []
            if ids is not None:
                cv.aruco.drawDetectedMarkers(frame, corners, ids)
                for c, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    centre = marker_centre_px(c.reshape(4, 2), cam.K, cam.dist)
                    x, y, _ = cam.pixel_to_arm(*centre, undistorted=True)
                    reach, pan = cam.to_polar(x, y)
                    text = f"id {marker_id}: x={x:5.1f} y={y:5.1f} cm  reach {reach:4.1f} pan {pan:5.1f}"
                    if marker_id in MARKER_POSITIONS_CM:
                        mx, my = MARKER_POSITIONS_CM[marker_id]
                        text += f"  err {math.hypot(x - mx, y - my) * 10:.1f}mm"
                    lines.append(text)
                    u, v = c.reshape(4, 2)[0].astype(int)
                    cv.putText(frame, str(marker_id), (u, v - 8), font, 0.6, (0, 255, 255), 2)
            for k, text in enumerate(sorted(lines)):
                cv.putText(frame, text, (10, 20 + 20 * k), font, 0.45, (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow("markers (arm frame)", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("p"):
                print("\n".join(sorted(lines)) + "\n" + "-" * 40)
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def draw_overlay(frame, cam, step=5, extent=(0, 35, -25, 25)):
    """Arm-frame grid, reach band, origin and shoulder point, drawn through the
    real (distorted) lens so lines bend exactly as the image does."""
    x0, x1, y0, y1 = extent
    for x in range(x0, x1 + 1, step):
        pts = cam.arm_to_pixel([[x, y, 0] for y in np.linspace(y0, y1, 40)])
        cv.polylines(frame, [pts.astype(np.int32)], False, (255, 200, 0), 1)
    for y in range(y0, y1 + 1, step):
        pts = cam.arm_to_pixel([[x, y, 0] for x in np.linspace(x0, x1, 40)])
        cv.polylines(frame, [pts.astype(np.int32)], False, (255, 200, 0), 1)
    for r in REACH_BAND_CM:
        pts = cam.arm_to_pixel([[r * math.cos(t), r * math.sin(t), 0]
                                for t in np.linspace(-math.pi / 2, math.pi / 2, 60)])
        cv.polylines(frame, [pts.astype(np.int32)], False, (0, 165, 255), 2)
    base, shoulder = cam.arm_to_pixel([[0, 0, 0], [0, 0, SHOULDER_HEIGHT_CM]]).astype(int)
    cv.drawMarker(frame, tuple(base), (0, 0, 255), cv.MARKER_CROSS, 18, 2)
    cv.line(frame, tuple(base), tuple(shoulder), (0, 0, 255), 2)
    cv.circle(frame, tuple(shoulder), 5, (0, 0, 255), -1)


def cmd_validate(_):
    """Grid lines should sit on a ruled sheet; the red dot should sit on the
    shoulder joint. Click any point on the table to read its arm coords."""
    cam = CameraToArm()
    cap = open_camera()
    clicks = []

    def on_click(event, u, v, *_):
        if event == cv.EVENT_LBUTTONDOWN:
            x, y, _ = cam.pixel_to_arm(u, v)
            reach, pan = cam.to_polar(x, y)
            band = "IN" if REACH_BAND_CM[0] <= reach <= REACH_BAND_CM[1] else "OUT OF"
            print(f"px ({u},{v}) -> x={x:6.2f} y={y:6.2f} cm | reach {reach:5.2f} cm "
                  f"pan {pan:6.1f} deg | {band} reach band")
            clicks.append((u, v))

    cv.namedWindow("validate")
    cv.setMouseCallback("validate", on_click)
    print("Click table points to read arm coords. q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_overlay(frame, cam)
            for u, v in clicks[-10:]:
                cv.circle(frame, (u, v), 4, (0, 0, 255), -1)
            cv.imshow("validate", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def cmd_check(_):
    """Run at pipeline start-up: has the camera been knocked?"""
    cam = CameraToArm()
    cap = open_camera()
    try:
        corners, _ = detect_corners(cap, make_detector(), n_frames=10)
    finally:
        cap.release()
    centres = {i: marker_centre_px(c, cam.K, cam.dist) for i, c in corners.items()}
    shift = cam.drift_px(centres)
    if shift > DRIFT_WARN_PX:
        print(f"DRIFT: markers moved {shift:.1f}px since calibration - run `calibrate`")
        sys.exit(1)
    print(f"OK: max marker shift {shift:.1f}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["calibrate", "markers", "validate", "check"])
    args = parser.parse_args()
    {"calibrate": cmd_calibrate, "markers": cmd_markers,
     "validate": cmd_validate, "check": cmd_check}[args.command](args)