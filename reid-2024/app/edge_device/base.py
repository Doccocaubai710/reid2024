import json
import os
import time
import traceback
import uuid
import warnings
from typing import Generator, Union

import cv2
import imagezmq
import numpy as np
import torch
from rich.console import Console
from ultralytics import YOLOv10

from app.utils import constants
from app.utils.utils import compute_area, non_max_suppression

console = Console()


warnings.filterwarnings("ignore")


class BaseEdgeDevice:
    def __init__(self, source: str):
        """
        Base class for edge devices

        :param source: Source of the input data. Accept:

            - Video: "video.mp4"
            - Stream: "rtsp://example.com/media.mp4"
            - Folder: "folder/" (containing videos)
        """
        self.source = source
        self.model = self._load_yolo_model()

    def _load_yolo_model(self) -> YOLOv10:
        try:
            model_path = "/home/aidev/workspace/vanh/reid-2024/app/assets/models/detect.pt"
            model = YOLOv10(model_path)
            console.print(
                "[bold cyan]YOLO[/bold cyan] detection model loaded [bold green]successfully[/bold green] :vampire:"
            )
            return model
        except Exception as e:
            console.print(
                f"[bold red]Error[/bold red] when loading YOLO model: {traceback.format_exc()}"
            )
            raise e

    def _read_video(self) -> Generator:
        try:
            cap = cv2.VideoCapture(self.source)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                yield cap, frame
        except Exception as e:
            console.print(
                f"[bold red]Error[/bold red] when open video file: {traceback.format_exc()}"
            )
            raise e
        finally:
            cap.release()

    def _read_rtsp(self) -> Generator:
        NotImplementedError("Source RTSP stream has not been supported yet")

    def _read_folder(self) -> Generator:
        raise NotImplementedError("Source folder has not been supported yet")

    def read_source(self) -> Generator:
        """
        Convert the input source to frames, then yield them
        """

        if not self.source or not isinstance(self.source, str):
            raise ValueError("Invalid source. Source must be a string")

        if self.source.startswith("rtsp://"):
            return self._read_rtsp()
        elif os.path.isfile(self.source):
            return self._read_video()
        elif os.path.isdir(self.source):
            return self._read_folder()
        else:
            raise ValueError(
                f"Invalid source: {self.source}. Must be a valid video path, stream url, or folder path."
            )

    def _post_process_detect_result(
    self, results: YOLOv10, height: int, width: int
) -> list:
        """
        Process detection results to only include body detections
        """
        if results:
            # Keep only "person" class, remove "face" from classes
            detections = non_max_suppression(results, ["person","face"], 0.25, 0.7)  # Remove "face"
            if detections.numel() > 0:
                boxes = detections[:, :4].cpu().numpy()
                print(f"Number of boxes:{len(boxes)}")
                total_area = sum([compute_area(box) for box in boxes])
                num_boxes = len(boxes)
                mean_area = total_area / num_boxes
                sum_criterion = (
                    (mean_area / (height * width)) + (0.1 * num_boxes)
                ) * 0.1
                criterion = sum_criterion
                boxes = boxes.astype(np.int32)
                boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
                confidences = detections[:, 4].cpu().numpy()
                class_ids = detections[:, 5].cpu().numpy()
                frame_limit = 3

        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)
            criterion = 0.01
            frame_limit = 15

        return boxes, confidences, class_ids, criterion, frame_limit

    def detect(self, idx: int, frame: np.ndarray):
        try:
            height, width = frame.shape[:2]
            # Perform inference
            with torch.no_grad():
                results = self.model.predict(frame, conf=0.25)[0]
            bboxes_xywh, scores, class_ids, criterion, frame_limit = (
                self._post_process_detect_result(results, height, width)
            )
            print(len(bboxes_xywh))
            output = []
            bodys = []
            # Remove face variable and processing

            # Only process body detections
            if len(bboxes_xywh) > 0:
                bboxes = bboxes_xywh.tolist()
                scores = scores.tolist()
                class_ids = class_ids.tolist()
                

                for bbox, score, class_id in zip(bboxes, scores, class_ids):
                    print(class_id)
                    if class_id == 0:  # Assuming class_id 0 is for bodies
                        bodys.append(
                            {"bbox": bbox, "score": score, "class_id": int(class_id)}
                        )

                output.extend(bodys)
                print(output)
                
            return (
                output,
                criterion,
                frame_limit,
            )
        except Exception:
            console.print(
                f"[bold red]Error[/bold red] when detecting frame {idx}: {traceback.format_exc()}"
            )
            return []

    @staticmethod
    def check_inside(box_a: list, box_b: list) -> bool:
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        if x1_a >= x1_b and y1_a >= y1_b and x2_a <= x2_b and y2_a <= y2_b:
            return True

        return False

    def okay_to_run_detect(
        self,
        cur_frame: np.ndarray,
        prev_frame: Union[np.ndarray, None],
        frame_count: int,
        criterion: int,
        frame_limit: int,
    ) -> bool:
        """
        Check if it's okay to run detection on the current frame

        :params:
            cur_frame: current frame
            prev_frame: previous frame
            frame_count: number of frames since last detection

        :return:
            True if it's okay to run detection, False otherwise
        """
        # # Convert to gray
        # height, width = cur_frame.shape[:2]
        # cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        # # Calculate the percentage of changed pixels between the current frame and the previous frame
        # if prev_frame is not None:
        #     # prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        #     frame_diff = cv2.absdiff(cur_gray, prev_frame)
        #     _, thresh = cv2.threshold(frame_diff, 150, 255, cv2.THRESH_BINARY)
        #     non_zero_count = np.count_nonzero(thresh)
        #     total_pixels = cur_gray.shape[0] * cur_gray.shape[1]

        #     change_pct = (non_zero_count / total_pixels) * 100
        # else:
        #     change_pct = 0
        #     frame_count = 0

        # if change_pct > criterion or frame_count == frame_limit or prev_frame is None:
        #     return True
        # else:
        #     return False
        return True

    def init_sender(self) -> bool:
        try:
            self.sender = imagezmq.ImageSender(
                connect_to=constants.SERVER_ADDR, REQ_REP=False
            )
        except Exception:
            console.print(
                f"[bold red]Error[/bold red] when initializing Image Sender: {traceback.format_exc()}"
            )
            return False
        else:
            console.print(
                "[bold cyan]Image Sender[/bold cyan] initialized [bold green]successfully[/bold green] :vampire:"
            )
            return True

    def get_start_stop_msg(
        self,
        cap: cv2.VideoCapture,
        is_start: bool,
        session_id: str,
        summary: Union[dict, None] = None,
    ) -> tuple:
        if is_start:
            msg = json.dumps(
                {
                    "session_id": session_id,
                    "status": "start",
                    "metadata": {
                        "source": self.source,
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "length": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                        "shape": (
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        ),
                    },
                }
            )

            return msg, np.zeros((640, 480, 3), np.uint8)
        else:
            return json.dumps(
                {"session_id": session_id, "status": "end", "summary": summary}
            ), np.zeros((640, 480, 3), np.uint8)

    def run(self):
        if not self.init_sender():
            return

        # Warmup the server
        time.sleep(2)

        prev_frame = None
        frame_count = 0
        skipped_frames = 0
        criterion = 0.04
        frame_limit = 5
        time_start = time.time()
        session_id = uuid.uuid4().hex
        cap = cv2.VideoCapture(self.source)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.sender.send_image(
            *self.get_start_stop_msg(cap, is_start=True, session_id=session_id)
        )

        # Warmup the server
        time.sleep(2)

        for idx, (cap, frame) in enumerate(self.read_source()):
            
            metadata = {
                "frame": idx,
                "source": self.source,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
            }
            if not self.okay_to_run_detect(
                frame, prev_frame, frame_count, criterion, frame_limit
            ):
                frame_count += 1
                msg = json.dumps(
                    {
                        "session_id": session_id,
                        "status": "running",
                        "metadata": metadata,
                        "is_skipped": True,
                        "detections": [],
                    }
                )
                # prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                skipped_frames += 1
                self.sender.send_image(msg, frame)

            else:
                output, criterion, frame_limit = self.detect(idx, frame)
                frame_count = 0
                msg = json.dumps(
                    {
                        "session_id": session_id,
                        "status": "running",
                        "metadata": metadata,
                        "is_skipped": False,
                        "detections": output,
                    }
                )
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.sender.send_image(msg, frame)
            print(f"frame_number: {idx}")
        # End the video
        self.sender.send_image(
            *self.get_start_stop_msg(
                cap,
                is_start=False,
                session_id=session_id,
                summary={
                    "time_elapsed": time.time() - time_start,
                    "skipped_frames": skipped_frames,
                    "total_frames": int(total_frames),
                },
            )
        )

        cap.release()
