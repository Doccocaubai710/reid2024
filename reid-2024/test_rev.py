import cv2
import zmq
import base64
import numpy as np
from datetime import datetime

RECEIVER_PORT = "5555"
SAVE_PATH = f"received_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind(f"tcp://*:{RECEIVER_PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages

print("Waiting for video frames... Saving to:", SAVE_PATH)

writer = None
fps = 30

try:
    while True:
        # 🟢 Receive JSON message
        msg = socket.recv_json()
        print("Received:", msg)
        exit()

        jpg_buffer = base64.b64decode(msg["image"])
        np_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Warning: Failed to decode image.")
            continue

        # 🟢 Init writer once with correct frame size
        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))
            print(f"Writer initialized: {width}x{height}")

        writer.write(frame)
        

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    if writer:
        writer.release()
        print("Video saved to:", SAVE_PATH)
    socket.close()
    context.term()
