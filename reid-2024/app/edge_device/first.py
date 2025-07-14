import sys
sys.path.append("/home/aidev/workspace/reid/Thesis/reid-2024/")
from app.edge_device.base import BaseEdgeDevice

first_device = BaseEdgeDevice(
    source="/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/videos/cam1.mp4"
)


# for idx, frame in enumerate(first_device.read_source()):
#     print(f"Frame {idx}, shape: {frame.shape}")
#     print(first_device.detect(idx, frame))

first_device.run()
