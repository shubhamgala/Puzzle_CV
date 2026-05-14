import cv2
import mediapipe as mp
import math
import os
import urllib.request


# Hand connections (landmark index pairs) — mirrors mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm knuckles
]

# Fingertip landmark indices
FINGERTIPS = [4, 8, 12, 16, 20]


class HandTracker:
    def __init__(self):
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("[HandTracker] Downloading hand_landmarker.task model...")
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                model_path
            )
            print("[HandTracker] Download complete.")

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.timestamp = 0

    def find_hands(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.results = self.hands.detect_for_video(mp_image, self.timestamp)
        self.timestamp += 33  # ~30 fps

    def draw_hands(self, frame):
        """Draw hand skeleton using pure OpenCV — no mp.solutions required."""
        if not (self.results and self.results.hand_landmarks):
            return

        h, w, _ = frame.shape

        for hand_landmarks in self.results.hand_landmarks:
            # Convert normalised coords → pixel coords
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            # Draw connections
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (80, 200, 255), 2)

            # Draw all landmark dots
            for idx, pt in enumerate(pts):
                color = (255, 255, 255) if idx in FINGERTIPS else (0, 180, 255)
                radius = 6 if idx in FINGERTIPS else 4
                cv2.circle(frame, pt, radius, color, -1)
                cv2.circle(frame, pt, radius, (0, 0, 0), 1)  # thin black border

    def get_pinch(self):
        """Detect pinch gesture (thumb tip ↔ index tip). Returns (pinch, x, y)."""
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                x1, y1 = hand_landmarks[4].x, hand_landmarks[4].y   # thumb tip
                x2, y2 = hand_landmarks[8].x, hand_landmarks[8].y   # index tip

                dist = math.hypot(x2 - x1, y2 - y1)

                if dist < 0.05:
                    return True, x2, y2

            # No pinch — return index tip of first hand as pointer
            hand_landmarks = self.results.hand_landmarks[0]
            return False, hand_landmarks[8].x, hand_landmarks[8].y

        return False, 0, 0

    def get_index_pos(self):
        """Get index finger tip of first detected hand. Returns (detected, x, y)."""
        if self.results and self.results.hand_landmarks:
            hand_landmarks = self.results.hand_landmarks[0]
            return True, hand_landmarks[8].x, hand_landmarks[8].y

        return False, 0, 0

    def get_two_hand_indices(self):
        """Get index finger tips of two hands. Returns (two_detected, p1, p2)."""
        points = []

        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                points.append((hand_landmarks[8].x, hand_landmarks[8].y))

        if len(points) >= 2:
            return True, points[0], points[1]

        return False, (0, 0), (0, 0)

    def get_two_hand_positions(self):
        """Get index finger positions of all detected hands as a list."""
        points = []

        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                points.append((hand_landmarks[8].x, hand_landmarks[8].y))

        return points