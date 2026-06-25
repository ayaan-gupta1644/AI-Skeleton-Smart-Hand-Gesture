import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks                import python as mp_python
from mediapipe.tasks.python         import vision as mp_vision
from mediapipe.tasks.python.vision  import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


# Hand skeleton connections (21-landmark topology)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm knuckle arch
]

# Landmark indices
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

FINGER_COLOURS = [
    (0,   165, 255),   # orange  – thumb
    (0,   255, 100),   # green   – index
    (255, 100,   0),   # blue    – middle
    (200,   0, 200),   # magenta – ring
    (0,   200, 255),   # yellow  – pinky
]

WRIST_COLOUR = (255, 255, 255)


# Landmark helpers

def lm_to_px(landmark, w, h):
    """Convert a NormalizedLandmark to (x_px, y_px, z)."""
    return int(landmark.x * w), int(landmark.y * h), landmark.z


def classify_side(handedness_list):
    """Return 'left' or 'right' from a mediapipe.tasks handedness list."""
    if handedness_list:
        return handedness_list[0].category_name.lower()
    return "unknown"


# Drawing helpers

def lerp_colour(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_3d_line(img, p1, p2, colour, z1=0.0, z2=0.0,
                 max_thickness=8, min_thickness=1):
    """Segmented line whose thickness and brightness encode depth."""
    segments = 20
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        x0 = int(p1[0] + (p2[0] - p1[0]) * t0)
        y0 = int(p1[1] + (p2[1] - p1[1]) * t0)
        x1 = int(p1[0] + (p2[0] - p1[0]) * t1)
        y1 = int(p1[1] + (p2[1] - p1[1]) * t1)
        z     = z1 + (z2 - z1) * ((t0 + t1) / 2.0)
        depth = np.clip(1.0 + z * 8, 0.1, 1.0)
        thick = max(min_thickness, int(max_thickness * depth))
        col   = lerp_colour(colour, (255, 255, 255), 1.0 - depth)
        cv2.line(img, (x0, y0), (x1, y1), col, thick, lineType=cv2.LINE_AA)


def draw_3d_polygon(img, points_z, colour, alpha=0.18):
    pts     = np.array([(p[0], p[1]) for p in points_z], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_glowing_circle(img, center, radius, colour, thickness=2):
    glow = tuple(min(255, c + 80) for c in colour)
    cv2.circle(img, center, radius + 4, glow,   1,         cv2.LINE_AA)
    cv2.circle(img, center, radius,     colour, thickness,  cv2.LINE_AA)


# Per-hand skeleton

def draw_hand_skeleton(img, landmarks, w, h, hand_colour=(80, 200, 255)):
    for (a, b) in HAND_CONNECTIONS:
        x1, y1, z1 = lm_to_px(landmarks[a], w, h)
        x2, y2, z2 = lm_to_px(landmarks[b], w, h)
        draw_3d_line(img, (x1, y1), (x2, y2), hand_colour, z1, z2,
                     max_thickness=4, min_thickness=1)
    for lm in landmarks:
        x, y, z = lm_to_px(lm, w, h)
        depth = np.clip(1.0 + z * 8, 0.2, 1.0)
        r = max(3, int(8 * depth))
        draw_glowing_circle(img, (x, y), r, hand_colour, thickness=2)


# Cross-hand 3D connections

def draw_finger_connections(img, lm_left, lm_right, w, h):
    tip_left  = [lm_to_px(lm_left[i],  w, h) for i in FINGER_TIPS]
    tip_right = [lm_to_px(lm_right[i], w, h) for i in FINGER_TIPS]

    # Semi-transparent face fills between adjacent finger pairs
    for i in range(len(FINGER_TIPS) - 1):
        face = [tip_left[i], tip_left[i+1], tip_right[i+1], tip_right[i]]
        col  = lerp_colour(FINGER_COLOURS[i], FINGER_COLOURS[i+1], 0.5)
        draw_3d_polygon(img, face, col, alpha=0.12)

    # Fingertip ↔ fingertip bridges
    for pl, pr, col in zip(tip_left, tip_right, FINGER_COLOURS):
        draw_3d_line(img, pl[:2], pr[:2], col, pl[2], pr[2],
                     max_thickness=6, min_thickness=2)

    # Ring around each hand's own fingertips
    for tips in (tip_left, tip_right):
        for i in range(len(tips)):
            j   = (i + 1) % len(tips)
            col = lerp_colour(FINGER_COLOURS[i], FINGER_COLOURS[j], 0.5)
            draw_3d_line(img, tips[i][:2], tips[j][:2], col,
                         tips[i][2], tips[j][2],
                         max_thickness=3, min_thickness=1)

    # Wrist–wrist connection
    wl = lm_to_px(lm_left[0],  w, h)
    wr = lm_to_px(lm_right[0], w, h)
    draw_3d_line(img, wl[:2], wr[:2], WRIST_COLOUR, wl[2], wr[2],
                 max_thickness=4, min_thickness=1)

    # Glowing fingertip spheres
    for tips in (tip_left, tip_right):
        for pt, col in zip(tips, FINGER_COLOURS):
            draw_glowing_circle(img, (pt[0], pt[1]), 7, col, thickness=2)
            cv2.circle(img, (pt[0], pt[1]), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Midpoint finger labels
    for pl, pr, name, col in zip(tip_left, tip_right, FINGER_NAMES, FINGER_COLOURS):
        mx = (pl[0] + pr[0]) // 2
        my = (pl[1] + pr[1]) // 2 - 14
        cv2.putText(img, name, (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# HUD

def draw_hud(img, hands_detected, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    status = "3D Finger Bridge Active" if hands_detected == 2 else "Waiting for 2 hands..."
    cv2.putText(img, f"Hands: {hands_detected}  |  {status}", (10, 27),
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

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Hand 3D Tracker started — press Q to quit.")
    print("Requires: hand_landmarker.task in the same folder.")
    print("Download: https://storage.googleapis.com/mediapipe-models/"
          "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

    timestamp_ms = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 33      # ~30 fps clock
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            lm_by_side    = {}
            hands_detected = 0

            if result.hand_landmarks:
                hands_detected = len(result.hand_landmarks)

                for i, landmarks in enumerate(result.hand_landmarks):
                    side = classify_side(
                        result.handedness[i] if result.handedness else []
                    )
                    lm_by_side[side] = landmarks
                    col = (80, 200, 255) if side == "right" else (255, 160, 60)
                    draw_hand_skeleton(frame, landmarks, w, h, hand_colour=col)

                if "left" in lm_by_side and "right" in lm_by_side:
                    draw_finger_connections(
                        frame,
                        lm_by_side["left"],
                        lm_by_side["right"],
                        w, h,
                    )

            draw_hud(frame, hands_detected, w, h)
            cv2.imshow("Hand 3D Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracker stopped.")


if __name__ == "__main__":
    main()