YOLO_MODEL = "./app/assets/models/detect.pt"
PLATFORM = "cuda"  # or "cpu"
SERVER_ADDR = "tcp://localhost:5555"

# Feature extraction
MODEL_PATH = "/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/models/model.pth.tar-150"
IMG_SIZE = (128, 64)
MAX_BATCH_SIZE = 4
BATCH_PROCESSING_SIZE = 20
THREADS = 10
ASYNC_MODE = False
TIME_TO_LIVE = 100  # 100 frames

# Draw color
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
THICKNESS = 2

# Skip frame
FRAME_DIFF_THRESHOLD = 0.04
MAX_FRAMES_FROM_LAST_DETECT = 5
