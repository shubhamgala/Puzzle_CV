import cv2

print("Testing with MSMF backend...")
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    opened = cap.isOpened()
    print(f"  Camera index {i} (MSMF): {opened}")
    if opened:
        ret, frame = cap.read()
        shape = frame.shape if ret else "N/A"
        print(f"    -> Frame read: {ret}, shape: {shape}")
    cap.release()

print("\nTesting with default backend (no flag)...")
for i in range(4):
    cap = cv2.VideoCapture(i)
    opened = cap.isOpened()
    print(f"  Camera index {i} (default): {opened}")
    if opened:
        ret, frame = cap.read()
        shape = frame.shape if ret else "N/A"
        print(f"    -> Frame read: {ret}, shape: {shape}")
    cap.release()

print("Done.")
