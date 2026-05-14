import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # suppress TF C++ logs
os.environ['GLOG_minloglevel'] = '2'               # suppress MediaPipe INFO/WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'         # suppress oneDNN messages

# ================= CAMERA CONFIG =================
# Set PHONE_IP to your phone's IP to use phone as webcam (e.g. "192.168.0.117:8080")
# Set PHONE_IP = None to use the laptop webcam (default)
PHONE_IP   = None    # <-- set to phone IP string to use phone, None for laptop camera
PHONE_PORT = 8080

import cv2
import time
import random
from hand_tracker import HandTracker
from puzzle import Puzzle


def open_laptop_camera():
    """Try multiple indices and backends to find a working laptop camera."""
    backends = [
        (cv2.CAP_ANY,   "default"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF,  "MSMF"),
    ]
    for idx in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[Camera] Found laptop camera at index={idx} backend={name}")
                    return cap
            cap.release()
    return None


# ================= CAMERA =================
if PHONE_IP:
    # Strip port from PHONE_IP if user accidentally included it (e.g. "192.168.0.117:8080")
    _ip_clean = PHONE_IP.split(":")[0]
    _port = int(PHONE_IP.split(":")[1]) if ":" in PHONE_IP else PHONE_PORT
    stream_url = f"http://{_ip_clean}:{_port}/video"
    print(f"[Camera] Connecting to phone stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[ERROR] Could not connect to phone stream: {stream_url}")
        print("  -> Make sure phone and laptop are on the SAME WiFi network.")
        print("  -> Make sure the IP Webcam app is running on your phone.")
        exit(1)
else:
    print("[Camera] Searching for laptop webcam...")
    cap = open_laptop_camera()
    if cap is None:
        print("[ERROR] No laptop webcam found!")
        print("  -> Make sure no other app (Zoom, Teams, browser) is using the camera.")
        print("  -> Check: Windows Settings > Privacy & Security > Camera > enable access.")
        print("  -> Or set PHONE_IP at the top of main.py to use your phone instead.")
        exit(1)

print("[Camera] Stream opened successfully!")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Live Puzzle", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Puzzle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

tracker = HandTracker()
puzzle = Puzzle(3)

mode = "camera"

# selection box
sel_x1 = sel_y1 = sel_x2 = sel_y2 = None

# states
start_time = None
end_time = None
solved = False

prev_pinch = False
dragging = False

# smoothing
smooth_x, smooth_y = 0, 0
alpha = 0.2

# trail
trail_points = []

# shuffle — last_shuffle must be outside the loop so the timer persists across frames
shuffling = False
shuffle_start = 0
last_shuffle = 0  # FIX: moved outside main loop so the 0.05s throttle works correctly


# ================= HELPERS =================
def inside_box(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h
    return x1 <= fx <= x2 and y1 <= fy <= y2


def to_local(px, py, x1, y1, x2, y2, w, h):
    fx = px * w
    fy = py * h

    # Clamp the coordinates to the puzzle bounds instead of cancelling the drop entirely
    fx = max(x1, min(fx, x2))
    fy = max(y1, min(fy, y2))

    lx = (fx - x1) / (x2 - x1)
    ly = (fy - y1) / (y2 - y1)

    # Prevent lx/ly from being >= 1.0 which would crash the get_index logic
    lx = min(lx, 0.999)
    ly = min(ly, 0.999)

    return lx, ly


def draw_grid(img, x1, y1, x2, y2, rows=3, cols=3):
    cell_w = (x2 - x1) // cols
    cell_h = (y2 - y1) // rows

    for i in range(1, cols):
        cv2.line(img, (x1 + i * cell_w, y1), (x1 + i * cell_w, y2), (0, 255, 0), 2)

    for i in range(1, rows):
        cv2.line(img, (x1, y1 + i * cell_h), (x2, y1 + i * cell_h), (0, 255, 0), 2)


# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    tracker.find_hands(frame)
    tracker.draw_hands(frame)

    pinch, px, py = tracker.get_pinch()
    detected, ix, iy = tracker.get_index_pos()
    two_hands, p1, p2 = tracker.get_two_hand_indices()

    h, w, _ = frame.shape

    # ================= CAMERA MODE =================
    if mode == "camera":

        if two_hands:
            x1 = int(p1[0] * w)
            y1 = int(p1[1] * h)

            x2 = int(p2[0] * w)
            y2 = int(p2[1] * h)

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            color = (0, 255, 0)
            if pinch:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if pinch and not prev_pinch:
                if abs(x2 - x1) > 100 and abs(y2 - y1) > 100:

                    sel_x1, sel_y1, sel_x2, sel_y2 = x1, y1, x2, y2

                    crop = frame[sel_y1:sel_y2, sel_x1:sel_x2]

                    if crop.size != 0:
                        puzzle.create(crop)

                        # Start shuffle
                        shuffling = True
                        shuffle_start = time.time()
                        last_shuffle = 0  # reset throttle timer for new shuffle

                        mode = "puzzle"
                        start_time = None  # start after shuffle ends

        prev_pinch = pinch

        # HUD: instructions
        cv2.putText(frame, "Use BOTH hands to frame area, then PINCH to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # DEBUG HUD — shows hand detection status so you know if tracking is working
        num_hands = len(tracker.results.hand_landmarks) if (tracker.results and tracker.results.hand_landmarks) else 0
        pinch_status = "PINCH DETECTED" if pinch else "no pinch"
        debug_color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
        cv2.putText(frame, f"Hands detected: {num_hands}  |  {pinch_status}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, debug_color, 2)
        if num_hands == 0:
            cv2.putText(frame, ">> Show your hands to the camera! <<",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        elif num_hands == 1:
            cv2.putText(frame, ">> Show BOTH hands to frame a region! <<",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("Live Puzzle", frame)

    # ================= PUZZLE MODE =================
    else:
        output = frame.copy()

        if sel_x1 is not None:
            puzzle_img = puzzle.combine()
            puzzle_img = cv2.resize(puzzle_img, (sel_x2 - sel_x1, sel_y2 - sel_y1))
            output[sel_y1:sel_y2, sel_x1:sel_x2] = puzzle_img

            draw_grid(output, sel_x1, sel_y1, sel_x2, sel_y2)

        # pointer smoothing
        if detected:
            cx = int(ix * w)
            cy = int(iy * h)

            cx = max(sel_x1, min(cx, sel_x2))
            cy = max(sel_y1, min(cy, sel_y2))

            smooth_x = int(alpha * cx + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * cy + (1 - alpha) * smooth_y)

            if pinch and inside_box(ix, iy, sel_x1, sel_y1, sel_x2, sel_y2, w, h):
                trail_points.append((smooth_x, smooth_y))
                if len(trail_points) > 15:
                    trail_points.pop(0)

        # ===== SHUFFLE =====
        # FIX: last_shuffle is now outside the loop; throttle works correctly
        if shuffling:
            if time.time() - shuffle_start < 1.8:
                if time.time() - last_shuffle > 0.05:
                    i = random.randint(0, len(puzzle.tiles) - 1)
                    j = random.randint(0, len(puzzle.tiles) - 1)
                    puzzle.swap(i, j)
                    last_shuffle = time.time()

                # Show "Shuffling..." overlay during animation
                cv2.putText(output, "Shuffling...",
                            (w // 3, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
            else:
                shuffling = False
                start_time = time.time()

        # ===== INTERACTION =====
        if not shuffling:
            sx = smooth_x / w
            sy = smooth_y / h

            if pinch and not prev_pinch:
                if inside_box(sx, sy, sel_x1, sel_y1, sel_x2, sel_y2, w, h):

                    local = to_local(sx, sy, sel_x1, sel_y1, sel_x2, sel_y2, w, h)

                    if local is not None:
                        lx, ly = local
                        idx = puzzle.get_index(lx, ly)

                        if idx is not None:
                            puzzle.selected = idx
                            dragging = True

            elif not pinch and prev_pinch:
                if dragging and puzzle.selected is not None:

                    local = to_local(sx, sy, sel_x1, sel_y1, sel_x2, sel_y2, w, h)

                    if local is not None:
                        lx, ly = local
                        idx2 = puzzle.get_index(lx, ly)
                        puzzle.swap(puzzle.selected, idx2)

                    trail_points = []
                    puzzle.selected = None
                    dragging = False

                    if not solved and puzzle.is_solved(puzzle.original_tiles):
                        solved = True
                        end_time = time.time()

        prev_pinch = pinch

        # ===== DRAW =====
        puzzle.draw_selected(output)

        for i in range(1, len(trail_points)):
            cv2.line(output, trail_points[i - 1], trail_points[i], (119, 221, 119), 3)

        if detected:
            cv2.circle(output, (smooth_x, smooth_y), 6, (255, 255, 255), -1)

        if start_time and not solved:
            elapsed = time.time() - start_time
            cv2.putText(output, f"{elapsed:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if solved:
            final_time = end_time - start_time

            cv2.putText(output, "SOLVED!",
                        (w // 3, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(output, f"Time: {final_time:.2f}s",
                        (w // 3, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Live Puzzle", output)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()