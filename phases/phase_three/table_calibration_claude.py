"""
table_calibration.py - overhead camera -> arm base frame (cm) via homography.
 
Phase 3a of the Vision-Guided Robotic Sorting Arm.
 
Usage:
    python table_calibration.py generate    # write printable ArUco marker PNGs
    python table_calibration.py calibrate   # detect markers, fit homography, save
    python table_calibration.py validate    # live view + 5 cm grid overlay; click to read coords
    python table_calibration.py check       # has the camera moved since calibration?
 
Runtime use in the pick pipeline:
    from table_calibration import TableMapper
    mapper = TableMapper()
    x, y = mapper.pixel_to_table(u, v, object_height_cm=1.25)
    reach, pan_deg = mapper.to_polar(x, y)
 
Coordinate frame (ARM BASE FRAME, centimetres):
    origin : shoulder_pan rotation axis, projected straight down onto the table
    +x     : straight ahead of the arm (pan = 0)
    +y     : to the arm's left
"""
 
import argparse
import json
import math
import sys
from pathlib import Path
 
import cv2
import numpy as np
 
# --------------------------------------------------------------------------
# Config - edit these
# --------------------------------------------------------------------------
CAMERA_INDEX = 1                 # overhead camera (wrist cam is probably 0)
FRAME_W, FRAME_H = 1280, 720
 
CALIB_DIR = Path("calibration")
CALIB_FILE = CALIB_DIR / "table_homography.json"
INTRINSICS_FILE = CALIB_DIR / "camera_intrinsics.npz"   # optional: K, dist
 
ARUCO_DICT = cv2.aruco.DICT_4X4_50
 
# Marker CENTRE positions in the arm base frame (cm).
# Placeholders - tape the markers down first, THEN measure and fill these in.
# Use at least 5 (6 is better): with exactly 4 the fit is exact and the
# residuals tell you nothing about accuracy.
# Keep them around the edge of the workspace, outside the arm's sweep,
# so they're never occluded.
MARKER_POSITIONS_CM = {
    0: (4.0, 18.0),
    1: (26.0, 18.0),
    2: (26.0, -18.0),
    3: (4.0, -18.0),
    4: (30.0, 0.0),
    5: (15.0, 24.0),
}
 
# Lens-to-table distance, measured with a tape (cm). Used for parallax correction.
CAMERA_HEIGHT_CM = 55.0
 
# Object heights (cm) - measure yours. Used for parallax correction.
OBJECT_HEIGHTS_CM = {
    "cube": 2.5,
    "cylinder": 3.0,
    "block": 2.0,
}
 
REACH_BAND_CM = (6.0, 21.0)      # usable band from the IK solver validation
DRIFT_WARN_PX = 3.0              # marker shift that means "recalibrate"
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
 
 
def load_intrinsics():
    """Return (K, dist) if a lens calibration exists, else (None, None)."""
    if INTRINSICS_FILE.exists():
        data = np.load(INTRINSICS_FILE)
        return data["K"], data["dist"]
    return None, None
 
 
def undistort_px(points, K, dist):
    """Undistort pixel points (keeps pixel units). No-op without intrinsics."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if K is None:
        return pts
    return cv2.undistortPoints(pts, K, dist, P=K)
 
 
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}")
    return cap
 
 
def grab_frame(cap, flush=5):
    """Discard buffered frames so we get a current one."""
    for _ in range(flush):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        sys.exit("Failed to read frame from camera")
    return frame
 
 
def detect_marker_centres(cap, detector, n_frames=20):
    """Average each marker's centre over n_frames to beat pixel noise.
 
    Returns ({marker_id: (u, v)}, last_frame). Markers seen in fewer than
    half the frames are dropped as unreliable.
    """
    acc = {}
    frame = None
    for _ in range(n_frames):
        frame = grab_frame(cap, flush=1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            acc.setdefault(int(i), []).append(c.reshape(4, 2).mean(axis=0))
    centres = {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= n_frames // 2}
    return centres, frame
 
 
# --------------------------------------------------------------------------
# Runtime mapper
# --------------------------------------------------------------------------
class TableMapper:
    """Pixel -> arm-base-frame table coordinates, with parallax correction."""
 
    def __init__(self, path=CALIB_FILE):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found - run `calibrate` first")
        cfg = json.loads(Path(path).read_text())
        self.H = np.array(cfg["H"], dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.cam_h = float(cfg["camera_height_cm"])
        self.nadir = np.array(cfg["nadir_cm"], dtype=np.float64)
        self.marker_px = {int(k): np.array(v) for k, v in cfg["marker_px"].items()}
        self.K, self.dist = load_intrinsics()
 
    def pixel_to_table(self, u, v, object_height_cm=0.0, undistorted=False):
        """Map an image point to (x, y) cm on the table plane.
 
        object_height_cm: height above the table of the point you're
        localising. The camera sees a raised point displaced away from the
        point directly below the lens, so without this a 2.5 cm cube near
        the edge of the workspace lands ~1 cm too far out.
          - top-face centre      -> full object height
          - segmentation-mask centroid -> start at ~half the height, then
            tune it against the validation grid
        """
        p = np.array([[[u, v]]], dtype=np.float64)
        if not undistorted:
            p = undistort_px(p, self.K, self.dist)
        xy = cv2.perspectiveTransform(p, self.H).reshape(2)
        if object_height_cm:
            scale = (self.cam_h - object_height_cm) / self.cam_h
            xy = self.nadir + (xy - self.nadir) * scale
        return float(xy[0]), float(xy[1])
 
    def table_to_pixel(self, x, y):
        """Inverse map (undistorted pixel coords) - used for overlays."""
        p = np.array([[[x, y]]], dtype=np.float64)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(2)
 
    @staticmethod
    def to_polar(x, y):
        """(x, y) cm -> (horizontal reach cm, shoulder_pan degrees).
 
        Reach is measured from the pan axis. If your IK measures reach from
        the shoulder_lift axis and that sits forward of the pan axis,
        subtract that offset before calling the IK solver.
        """
        return math.hypot(x, y), math.degrees(math.atan2(y, x))
 
    def drift_px(self, centres):
        """Max pixel shift of any marker versus calibration time."""
        shifts = [np.linalg.norm(centres[i] - p) for i, p in self.marker_px.items() if i in centres]
        return max(shifts) if shifts else float("inf")
 
 
# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_generate(args):
    """Write marker PNGs. Print at 100% scale, ~4 cm square, on matte paper."""
    out = CALIB_DIR / "markers"
    out.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for marker_id in MARKER_POSITIONS_CM:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, 400)
        img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(img, f"id {marker_id}", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.imwrite(str(out / f"marker_{marker_id}.png"), img)
    print(f"Wrote {len(MARKER_POSITIONS_CM)} markers to {out}/")
 
 
def cmd_calibrate(args):
    cap = open_camera()
    detector = make_detector()
    K, dist = load_intrinsics()
    centres, frame = detect_marker_centres(cap, detector)
    cap.release()
 
    ids = [i for i in MARKER_POSITIONS_CM if i in centres]
    missing = sorted(set(MARKER_POSITIONS_CM) - set(ids))
    if missing:
        print(f"WARNING: markers not detected: {missing}")
    if len(ids) < 4:
        sys.exit(f"Need at least 4 markers, found {len(ids)}")
    if len(ids) == 4:
        print("NOTE: only 4 markers - fit is exact, residuals will be ~0 and meaningless")
 
    src = undistort_px([centres[i] for i in ids], K, dist).reshape(-1, 2)
    dst = np.array([MARKER_POSITIONS_CM[i] for i in ids], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)   # least squares over all markers
    if H is None:
        sys.exit("Homography fit failed - check marker positions aren't collinear")
 
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs_mm = np.linalg.norm(proj - dst, axis=1) * 10.0
    print("\nPer-marker residual (mm):")
    for i, e in zip(ids, errs_mm):
        print(f"  id {i}: {e:5.2f}")
    print(f"  mean {errs_mm.mean():.2f}  max {errs_mm.max():.2f}")
    if errs_mm.max() > 5:
        print("  -> >5 mm: re-measure marker positions; if edge markers are worst, "
              "lens distortion is the likely cause (do intrinsic calibration)")
 
    # Point on the table directly below the lens (assumes camera points
    # straight down; principal point ~ image centre without intrinsics).
    if K is not None:
        cx, cy = K[0, 2], K[1, 2]
    else:
        cx, cy = frame.shape[1] / 2, frame.shape[0] / 2
    nadir = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float64), H).reshape(2)
 
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps({
        "H": H.tolist(),
        "camera_height_cm": CAMERA_HEIGHT_CM,
        "nadir_cm": nadir.tolist(),
        "frame_size": [frame.shape[1], frame.shape[0]],
        "marker_px": {str(i): centres[i].tolist() for i in ids},
        "residuals_mm": {str(i): float(e) for i, e in zip(ids, errs_mm)},
    }, indent=2))
 
    vis = frame.copy()
    for i in ids:
        u, v = centres[i].astype(int)
        cv2.circle(vis, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(CALIB_DIR / "calibration_frame.png"), vis)
    print(f"\nSaved {CALIB_FILE}  (camera nadir at x={nadir[0]:.1f}, y={nadir[1]:.1f} cm)")
 
 
def draw_grid(frame, mapper, step=5, extent=(0, 35, -25, 25)):
    """Overlay the table grid (every `step` cm) and the reach band."""
    x0, x1, y0, y1 = extent
    for x in range(x0, x1 + 1, step):
        a, b = mapper.table_to_pixel(x, y0), mapper.table_to_pixel(x, y1)
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (255, 200, 0), 1)
    for y in range(y0, y1 + 1, step):
        a, b = mapper.table_to_pixel(x0, y), mapper.table_to_pixel(x1, y)
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (255, 200, 0), 1)
    for r in REACH_BAND_CM:
        pts = [mapper.table_to_pixel(r * math.cos(t), r * math.sin(t))
               for t in np.linspace(-math.pi / 2, math.pi / 2, 60)]
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 165, 255), 2)
    origin = mapper.table_to_pixel(0, 0).astype(int)
    cv2.drawMarker(frame, tuple(origin), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
 
 
def cmd_validate(args):
    """Live view. Grid lines should sit on a ruled sheet; clicks print coords."""
    mapper = TableMapper()
    cap = open_camera()
    clicks = []
 
    def on_click(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = mapper.pixel_to_table(u, v, undistorted=True)
            reach, pan = mapper.to_polar(x, y)
            ok = REACH_BAND_CM[0] <= reach <= REACH_BAND_CM[1]
            print(f"px ({u},{v}) -> x={x:6.2f} y={y:6.2f} cm | reach {reach:5.2f} cm "
                  f"pan {pan:6.1f} deg | {'IN' if ok else 'OUT OF'} reach band")
            clicks.append((u, v))
 
    cv2.namedWindow("validate")
    cv2.setMouseCallback("validate", on_click)
    print("Click points to read table coords. q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mapper.K is not None:
            frame = cv2.undistort(frame, mapper.K, mapper.dist)
        draw_grid(frame, mapper)
        for u, v in clicks[-10:]:
            cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)
        cv2.imshow("validate", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
 
 
def cmd_check(args):
    """Run at pipeline start-up: has the camera been knocked?"""
    mapper = TableMapper()
    cap = open_camera()
    centres, _ = detect_marker_centres(cap, make_detector(), n_frames=10)
    cap.release()
    shift = mapper.drift_px(centres)
    if shift > DRIFT_WARN_PX:
        print(f"DRIFT: markers moved {shift:.1f}px since calibration - run `calibrate`")
        sys.exit(1)
    print(f"OK: max marker shift {shift:.1f}px")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "calibrate", "validate", "check"])
    args = parser.parse_args()
    {"generate": cmd_generate, "calibrate": cmd_calibrate,
     "validate": cmd_validate, "check": cmd_check}[args.command](args)