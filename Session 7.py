import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks                import python as mp_python
from mediapipe.tasks.python         import vision as mp_vision
from mediapipe.tasks.python.vision  import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# Pose skeleton connections (33-landmark topology)
POSE_CONNECTIONS = [
    # Face details
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (12, 14), (14, 16), # Right arm
    (11, 13), (13, 15), (15, 17), # Left arm
    (11, 23), (12, 24), (23, 24), # Shoulders & Hips box
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left Leg + Foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), # Right Leg + Foot
]

# Joint Highlights
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

TORSO_COLOUR = (255, 150, 0)   # Blue-ish cyan for torso base
LIMB_COLOUR  = (0, 255, 100)   # Green for fast limb tracking


# Landmark helpers

def lm_to_px(landmark, w, h):
    """Convert a NormalizedLandmark to (x_px, y_px, z, visibility)."""
    return int(landmark.x * w), int(landmark.y * h), landmark.z, landmark.visibility


# Drawing helpers

def lerp_colour(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_3d_line(img, p1, p2, colour, z1=0.0, z2=0.0,
                 max_thickness=10, min_thickness=2):
    """Segmented line whose thickness and brightness encode depth."""
    segments = 15
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        x0 = int(p1[0] + (p2[0] - p1[0]) * t0)
        y0 = int(p1[1] + (p2[1] - p1[1]) * t0)
        x1 = int(p1[0] + (p2[0] - p1[0]) * t1)
        y1 = int(p1[1] + (p2[1] - p1[1]) * t1)
        
        # Z mapping for depth effects (MediaPipe Pose Z is roughly relative)
        z     = z1 + (z2 - z1) * ((t0 + t1) / 2.0)
        depth = np.clip(1.0 - z, 0.2, 1.2)  
        
        thick = max(min_thickness, int(max_thickness * depth))
        col   = lerp_colour(colour, (255, 255, 255), np.clip(1.2 - depth, 0.0, 0.8))
        cv2.line(img, (x0, y0), (x1, y1), col, thick, lineType=cv2.LINE_AA)


def draw_3d_polygon(img, points, colour, alpha=0.15):
    pts     = np.array([(p[0], p[1]) for p in points], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glowing_circle(img, center, radius, colour, thickness=2):
    glow = tuple(min(255, c + 80) for c in colour)
    cv2.circle(img, center, radius + 4, glow,   1,         cv2.LINE_AA)
    cv2.circle(img, center, radius,     colour, thickness,  cv2.LINE_AA)


# Pose Skeleton Rendering

def draw_pose_skeleton(img, landmarks, w, h):
    # Extracted list of all points mapped to screen pixels
    pts_px = [lm_to_px(lm, w, h) for lm in landmarks]
    
    # 1. Overlay a semi-transparent HUD polygon for the Torso box (11, 12, 24, 23)
    if all(pts_px[i][3] > 0.5 for i in [11, 12, 24, 23]):
        torso_pts = [pts_px[11], pts_px[12], pts_px[24], pts_px[23]]
        draw_3d_polygon(img, torso_pts, TORSO_COLOUR, alpha=0.2)

    # 2. Draw Skeleton bones
    for (a, b) in POSE_CONNECTIONS:
        x1, y1, z1, vis1 = pts_px[a]
        x2, y2, z2, vis2 = pts_px[b]
        
        # Only draw lines if both landmarks pass the visibility metric thresholds
        if vis1 > 0.5 and vis2 > 0.5:
            # Use Torso colour for trunk, Limb colour for peripheral lines
            col = TORSO_COLOUR if (a in [11,12,23,24] and b in [11,12,23,24]) else LIMB_COLOUR
            draw_3d_line(img, (x1, y1), (x2, y2), col, z1, z2, max_thickness=6, min_thickness=1)

    # 3. Render specialized tracking nodes
    for idx in CORE_JOINTS:
        x, y, z, vis = pts_px[idx]
        if vis > 0.5:
            depth = np.clip(1.0 - z, 0.3, 1.2)
            r = max(4, int(9 * depth))
            draw_glowing_circle(img, (x, y), r, LIMB_COLOUR, thickness=2)
            cv2.circle(img, (x, y), max(2, int(3 * depth)), (255, 255, 255), -1, cv2.LINE_AA)


# HUD

def draw_hud(img, pose_detected, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    status = "Kinematics Active" if pose_detected else "Scanning for target..."
    cv2.putText(img, f"Pose System: {status}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Q = quit", (w - 90, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


# Main

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="pose_landmarker.task"
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Pose 3D Tracker started — press Q to quit.")
    print("Requires: pose_landmarker_full.task in the same folder.")

    timestamp_ms = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 33  # ~30 fps dynamic clock tracker
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_detected = False

            if result.pose_landmarks:
                pose_detected = True
                for pose_landmarks in result.pose_landmarks:
                    draw_pose_skeleton(frame, pose_landmarks, w, h)

            draw_hud(frame, pose_detected, w, h)
            cv2.imshow("Pose 3D Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracker stopped.")


if __name__ == "__main__":
    main()