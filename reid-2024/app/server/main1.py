import json
import logging
import os
import pickle
import time
import traceback
from datetime import datetime
from typing import List
import torch
import cv2
import imagezmq
import numpy as np
from rich.console import Console
import psutil
import resource
import gc

from app.deeputils.base_track import BaseTrack
# CHANGED: Import TrackTrack instead of ByteTracker
from app.tracker.trackers.tracker import Tracker  # TrackTrack implementation
from app.utils import const
from app.utils.benchmark import BenchmarkMultiThread

console = Console()
logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

def get_face_body_from_detection(image: np.ndarray, detections: list):
    """Process the detections to get body and face bounding boxes."""
    faces, bodys = [], []
    
    for detection in detections:
        x, y, w, h = list(map(lambda x: int(x), detection.get("bbox")))
        detection.update({"frame": image[y : y + h, x : x + w]})
        bodys.append(detection)
        
    return bodys

def convert_detections_to_tracktrack_format(detections):
    """Convert detections to TrackTrack expected format"""
    formatted_dets = []
    formatted_dets_95 = []  # High confidence detections
    
    for detection in detections:
        bbox = detection.get("bbox")
        score = detection.get("score")
        
        # TrackTrack expects detections in specific format
        # Usually [x1, y1, x2, y2, confidence] or similar
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        det_array = [x1, y1, x2, y2, score]
        formatted_dets.append(det_array)
        
        # High confidence detections (threshold 0.95)
        if score >= 0.65:
            formatted_dets_95.append(det_array)
    
    # Ensure arrays are always 2D, even when empty
    if len(formatted_dets) == 0:
        formatted_dets_array = np.empty((0, 5))  # Empty 2D array
    else:
        formatted_dets_array = np.array(formatted_dets)

    if len(formatted_dets_95) == 0:
        formatted_dets_95_array = np.empty((0, 5))  # Empty 2D array  
    else:
        formatted_dets_95_array = np.array(formatted_dets_95)

    return formatted_dets_array, formatted_dets_95_array

class TrackArgs:
    """Simple args class for TrackTracker configuration"""
    def __init__(self):
        # Detection thresholds
        self.det_thr = 0.35
        self.match_thr = 0.8
        
        # Track initialization
        self.init_thr = 0.7
        self.tai_thr = 0.55
        
        # Penalty terms for association
        self.penalty_p = 0.2
        self.penalty_q = 0.4
        self.reduce_step = 0.05
        
        # Track management
        self.max_time_lost = 30  # frames

class MainServer:
    def __init__(self):
        self.receiver = None
        self.video_writers = {}
        self.tracking_file = None
        self.bench_mark_data = {}
        self.log_step = [10, 40, 100, 200, 400]
        self.process = psutil.Process()
        
        # TrackTrack configuration
        self.args = TrackArgs()
        
    def check_system_limits(self):
        """Check various system limits"""
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            current_fds = self.process.num_fds()
            
            if current_fds > soft_limit * 0.9:
                console.print(f"[red]WARNING: File descriptors near limit![/red]")
        except Exception as e:
            console.print(f"Error checking system limits: {e}")
            
    def check_memory_usage(self):
        """Check memory usage"""
        memory = psutil.virtual_memory()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        console.print(f"Memory: {memory.percent}% | Process memory: {memory_mb:.1f} MB")
        return memory.percent, memory_mb
        
    def diagnose_frame_issue(self, frame_number, session_id, sessions_storage):
        """Comprehensive diagnostic for frame issues"""
        console.print(f"\n{'='*20} FRAME {frame_number} DIAGNOSTICS {'='*20}")
        
        # Check memory
        memory_percent, process_memory = self.check_memory_usage()
        
        # Check file descriptors
        self.check_system_limits()
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        
        # Check video writer status
        if session_id in sessions_storage:
            video_writer = sessions_storage[session_id]
            console.print(f"VideoWriter exists: {video_writer is not None}")
            if video_writer:
                try:
                    writer_status = video_writer.isOpened()
                    console.print(f"VideoWriter opened: {writer_status}")
                except AttributeError:
                    console.print("VideoWriter.isOpened() not available")
        
        console.print("="*60)
        
    def init_receiver(self) -> bool:
        try:
            import zmq
            import numpy as np
            import json
            import base64
            
            # Create ZMQ context and socket
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.SUB)
            
            # Bind to all interfaces on port 5555
            self.zmq_socket.bind("tcp://0.0.0.0:5555")
            
            # Subscribe to all messages
            self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Set socket options for better performance
            self.zmq_socket.setsockopt(zmq.RCVHWM, 10000)
            self.zmq_socket.setsockopt(zmq.RCVBUF, 32 * 1024 * 1024)
            self.zmq_socket.setsockopt(zmq.LINGER, 1000)
            
            # Create a custom receiver object with recv_image method
            class ZmqReceiver:
                def __init__(self, socket):
                    self.zmq_socket = socket
                    
                def recv_image(self):
                    """Receive an image with its identifier message."""
                    # Receive message identifier
                    packet = self.zmq_socket.recv_json()
                    msg = packet["meta"]
                    
                    # Extract image
                    jpg_buffer = base64.b64decode(packet["image"])
                    np_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                    return msg, frame
            
            # Create the receiver object
            self.receiver = ZmqReceiver(self.zmq_socket)
            
            console.print("[bold cyan]Image Receiver[/bold cyan] initialized [bold green]successfully[/bold green] with increased buffer capacity")
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error when initializing Image Receiver: {str(e)}[/bold red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def draw_detection(self, img, bboxes, scores, ids):
        """Draw tracking results with corner-style bounding boxes and consistent colors"""
        import random
        
        # Initialize color dictionary if not exists
        if not hasattr(self, 'id2color'):
            self.id2color = {}
        
        def get_color_for_id(track_id):
            if track_id not in self.id2color:
                random.seed(int(track_id))  # Keep stable color by ID
                color = tuple(random.randint(50, 255) for _ in range(3))  # Avoid dark colors
                self.id2color[track_id] = color
            return self.id2color[track_id]
        
        if len(bboxes) == 0:
            return img
            
        # Prepare track_output format: [x1, y1, x2, y2, track_id]
        track_output = []
        for bbox, score, track_id in zip(bboxes, scores, ids):
            track_output.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        track_output = np.array(track_output)
        
        color_text = (255, 255, 255)
        draw = img.copy()
        bboxes_array = track_output[:, :4]
        identities = track_output[:, 4]
        
        for bbox, id in zip(bboxes_array, identities):
            x1, y1, x2, y2 = map(int, bbox.round())
            index = int(id) if id is not None else 0
            color_box = get_color_for_id(index)
            
            # Rectangle bounding box
            cv2.rectangle(draw, (x1, y1), (x2, y2), color_box, thickness=1)
            
            # Corner line weight
            lw = 2
            len_line_w = 20
            len_line_h = 40
            
            # Make corners
            cv2.line(draw, (x1, y1), (x1 + len_line_w, y1), color_box, lw)
            cv2.line(draw, (x1, y1), (x1, y1 + len_line_h), color_box, lw)
            cv2.line(draw, (x2, y1), (x2 - len_line_w, y1), color_box, lw)
            cv2.line(draw, (x2, y1), (x2, y1 + len_line_h), color_box, lw)
            cv2.line(draw, (x1, y2), (x1 + len_line_w, y2), color_box, lw)
            cv2.line(draw, (x1, y2), (x1, y2 - len_line_h), color_box, lw)
            cv2.line(draw, (x2, y2), (x2 - len_line_w, y2), color_box, lw)
            cv2.line(draw, (x2, y2), (x2, y2 - len_line_h), color_box, lw)
            
            # ID text config
            label = f"Human {index}"
            font_scale = 0.4
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            padding = 3
            y_offset = 5  # Distance between bbox and ID text
            text_bg_topleft = (x1, y1 - th - 2 * padding - y_offset)
            text_bg_bottomright = (x1 + tw + 2 * padding, y1 - y_offset)
            
            # Draw background rectangle
            cv2.rectangle(draw, text_bg_topleft, text_bg_bottomright, color_box, -1)
            
            # Draw text
            text_org = (x1 + padding, y1 - padding - y_offset)
            cv2.putText(draw, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, font_thickness, cv2.LINE_AA)
        
        return draw
        
    def draw_detection_only(self, img, bboxes, scores, mask_alpha=0.3):
        """Draw detection boxes with confidence scores only (no tracking IDs)"""
        height, width = img.shape[:2]
        np.random.seed(0)
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(1, 3))
        mask_img = img.copy()
        det_img = img.copy()
        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)
        
        for bbox, score in zip(bboxes, scores):
            color = colors[0]
            bbox = np.array(bbox)
            x, y, w, h = bbox.astype(np.int64)
            
            # Draw rectangle using top-left point (x,y) and width/height (w,h)
            cv2.rectangle(det_img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(mask_img, (x, y), (x + w, y + h), color, -1)
            
            caption = f"detection {int(score*100)}%"
            
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=size,
                thickness=text_thickness,
            )

            th = int(th * 1.2)
            cv2.rectangle(det_img, (x, y), (x + tw, y - th), color, -1)
            cv2.rectangle(mask_img, (x, y), (x + tw, y - th), color, -1)
            cv2.putText(
                det_img,
                caption,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_img,
                caption,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    @property
    def current_time_str(self):
        return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    
    def handle_start_frame(self, fourcc: int, metadata: dict, session_id: str, sessions_storage: dict):
        """Handle the start frame of a session with diagnostic checks"""
        console.print(f"\n[bold cyan]START FRAME HANDLER[/bold cyan]")
        console.print(f"Session ID: {session_id}")
        console.print(f"Sessions storage keys: {list(sessions_storage.keys())}")
        console.print(f"Current save_dir: {getattr(self, 'save_dir', None)}")
        
        if session_id not in sessions_storage:
            save_dir = f"trash/{self.current_time_str}-{session_id}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{os.path.basename(metadata.get('source'))}.avi"
            console.print(f"[yellow]Saving video to: {os.path.abspath(save_path)}[/yellow]")
            
            try:
                self.tracking_file = open(f"{save_dir}/tracking_results.txt", "w")
                console.print("[green]Tracking file opened successfully[/green]")
            except Exception as e:
                console.print(f"[red]Error opening tracking file: {e}[/red]")
                
            video_writer = cv2.VideoWriter(
                save_path,
                fourcc,
                metadata.get("fps"),
                (
                    int(metadata.get("shape")[0]),
                    int(metadata.get("shape")[1]),
                ),
            )
            self.video_writers[session_id] = video_writer
            self.benchmark = BenchmarkMultiThread()
            self.benchmark.start_logging()
            self.step_ids = 0
            self.save_dir = save_dir            
            console.print(f"Save path for session [bold green]{session_id}[/bold green]: {save_path}")
                
    def run(self):
        ############################## INITIALIZATION ##############################
        if not self.init_receiver():
            console.print("[bold red] Error when initializing receiver [/bold red]")
            return
            
        frame_count = 0
        # CHANGED: Initialize TrackTracker instead of BYTETracker
        tracktrack = Tracker(self.args, vid_name="video_session", use_cmc=False)  # Disable CMC
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        sessions_storage = {}
        self.detection_video_writers = {}
        
        # Initial system check
        console.print("\n[bold cyan]Initial System Check[/bold cyan]")
        self.check_system_limits()
        self.check_memory_usage()
        
        while True:
            try:
                if frame_count % 100 == 0:
                    self.check_system_limits()
                    self.check_memory_usage()
                
                info, opencv_image = self.receiver.recv_image()
                console.print(f"Image received")
                info = json.loads(info)
                
                detections = info.get("detections")
                metadata = info.get("metadata")
                session_id = info.get("session_id")
                frame_count += 1
                
                if metadata:
                    frame_number = metadata.get("frame", [])
                    print(frame_number)
                else:
                    console.print("No metadata")
                logging.info(f"Frame_number: {frame_number}")
                
                status = info.get("status", "NO STATUS")
                console.print(f"Message status: '{status}'")
                
                if info.get("status") == "start":
                    console.print(f"[bold yellow][START][/bold yellow] frame:{frame_number} of session [bold cyan]{session_id}[/bold cyan]")
                    self.handle_start_frame(fourcc=fourcc, metadata=metadata, session_id=session_id, sessions_storage=sessions_storage)
                    
                    # Initialize detection-only video writer
                    if session_id not in self.detection_video_writers:
                        detection_save_path = f"{self.save_dir}/detection_only_{os.path.basename(metadata.get('source'))}.avi"
                        detection_writer = cv2.VideoWriter(
                            detection_save_path,
                            fourcc,
                            metadata.get("fps"),
                            (int(metadata.get("shape")[0]), int(metadata.get("shape")[1])),
                        )
                        self.detection_video_writers[session_id] = detection_writer
                        console.print(f"[green]Detection-only video writer created at {detection_save_path}[/green]")
                    
                    self.start_time = time.perf_counter()
                    
                elif info.get("status") == "running":
                    
                    if hasattr(self, 'step_ids'):
                        self.step_ids += 1
                    
                    if hasattr(self, 'step_ids') and self.step_ids in self.log_step:
                        if hasattr(self, 'benchmark'):
                            data = self.benchmark.calculate_averages()
                            console.print(f"Usage at step {self.step_ids}: {data}")
                            self.bench_mark_data[self.step_ids] = data
                    
                    # Regular memory cleanup
                    if frame_count % 50 == 0:
                        gc.collect()
                        memory_percent = psutil.virtual_memory().percent
                        console.print(f"[yellow]Memory usage: {memory_percent}%[/yellow]")
                        
                        if memory_percent > 85:
                            console.print(f"[red]HIGH MEMORY USAGE: {memory_percent}%[/red]")
                            gc.collect()
                    
                    console.print(f"[bold yellow][RUNNING][/bold yellow] frame:{frame_number} of session [bold cyan]{session_id}[/bold cyan]")
                    
                    # Get the video writers for the current session
                    video_writer = self.video_writers.get(session_id)
                    detection_writer = self.detection_video_writers.get(session_id)
                    
                    if video_writer is None:
                        console.print(f"[red]ERROR: No tracking video writer for session {session_id}[/red]")
                        continue
                    
                    if detection_writer is None:
                        console.print(f"[red]ERROR: No detection video writer for session {session_id}[/red]")
                        continue
                    
                    # Process detections for TrackTrack format
                    bodys = get_face_body_from_detection(opencv_image, detections)
                    _time = time.perf_counter()
                    
                    # Extract bboxes and scores for detection visualization
                    detection_boxes = []
                    detection_confidences = []
                    
                    for body in bodys:
                        if body.get("frame").tolist() == []:
                            continue
                        detection_boxes.append(body.get("bbox"))
                        detection_confidences.append(body.get("score"))
                    
                    detection_boxes = np.asarray(detection_boxes) if detection_boxes else np.array([]).reshape(0, 4)
                    detection_confidences = np.asarray(detection_confidences) if detection_confidences else np.array([])
                    
                    # Convert detections to TrackTrack format
                    tracktrack_dets, tracktrack_dets_95 = convert_detections_to_tracktrack_format(detections)
                    
                    console.print(f"Process detections time: {time.perf_counter() - _time}")
                    console.print(f"Found {len(tracktrack_dets)} detections")

                    # Create detection-only visualization (before tracking)
                    detection_img = opencv_image.copy()
                    if len(detection_boxes) > 0:
                        detection_img = self.draw_detection_only(
                            img=opencv_image.copy(), 
                            bboxes=detection_boxes, 
                            scores=detection_confidences
                        )
                    
                    # CHANGED: Use TrackTracker's update method
                    tracked_objects = tracktrack.update(tracktrack_dets, tracktrack_dets_95)
                    
                    # Extract tracking results
                    bboxes = []
                    scores = []
                    ids = []
                    
                    for track in tracked_objects:
                        # TrackTrack returns Track objects with different attributes
                        bbox = track.x1y1x2y2  # Get bounding box in [x1, y1, x2, y2] format
                        bboxes.append(bbox)
                        scores.append(track.score)
                        ids.append(track.track_id)
                    
                    console.print(f"Number of tracked objects: {len(tracked_objects)}")
                    
                    # Write tracking data to file
                    if len(bboxes) > 0:
                        try:
                            for i in range(len(bboxes)):
                                line = f"{frame_number},{ids[i]},{bboxes[i][0]},{bboxes[i][1]},{bboxes[i][2]},{bboxes[i][3]},{scores[i]}\n"
                                self.tracking_file.write(line)
                                self.tracking_file.flush()
                        except Exception as e:
                            console.print(f"[red]Error writing to tracking file: {e}[/red]")
                        
                        # Draw tracking results
                        result_img = self.draw_detection(img=opencv_image, bboxes=bboxes, scores=scores, ids=ids)
                    else:
                        result_img = opencv_image
                    
                    # Enhanced video writing with error checking
                    try:
                        # Add frame number text to both visualizations
                        text = f"Frame: {frame_number}"
                        cv2.putText(
                            result_img,
                            text,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                        )
                        
                        cv2.putText(
                            detection_img,
                            text,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            1,
                        )
                        
                        # Write both videos
                        video_writer.write(result_img)
                        detection_writer.write(detection_img)
                        
                        # Check write status near frame 999
                        if frame_number >= 995:
                            console.print(f"[green]Frame {frame_number} written successfully[/green]")
                            
                    except Exception as e:
                        console.print(f"[red]Error writing frame {frame_number}: {e}[/red]")
                        
                        # Try to recover video writers
                        try:
                            # Recover tracking writer
                            video_writer.release()
                            video_writer = cv2.VideoWriter(
                                f"{self.save_dir}/{os.path.basename(metadata.get('source'))}_tracking_recovery.avi",
                                fourcc,
                                metadata.get("fps"),
                                (int(metadata.get("shape")[0]), int(metadata.get("shape")[1])),
                            )
                            self.video_writers[session_id] = video_writer
                            
                            # Recover detection writer
                            detection_writer.release()
                            detection_writer = cv2.VideoWriter(
                                f"{self.save_dir}/{os.path.basename(metadata.get('source'))}_detection_recovery.avi",
                                fourcc,
                                metadata.get("fps"),
                                (int(metadata.get("shape")[0]), int(metadata.get("shape")[1])),
                            )
                            self.detection_video_writers[session_id] = detection_writer
                            
                            console.print("[yellow]Attempted to recover VideoWriters[/yellow]")
                        except Exception as e2:
                            console.print(f"[red]Failed to recover VideoWriters: {e2}[/red]")
                            
                elif info.get("status") == "end":
                    if hasattr(self, 'step_ids'):
                        self.step_ids += 1
                        
                        # Log the last step
                        if hasattr(self, 'benchmark'):
                            data = self.benchmark.calculate_averages()
                            console.print(f"Usage at step {self.step_ids}: {data}")
                            self.bench_mark_data[self.step_ids] = data
                    
                    # Close all video writers
                    video_writer = self.video_writers.pop(session_id, None)
                    if video_writer:
                        video_writer.release()
                    
                    detection_writer = self.detection_video_writers.pop(session_id, None)
                    if detection_writer:
                        detection_writer.release()
                    
                    if self.tracking_file:
                        self.tracking_file.close()
                    
                    # Save the benchmark data
                    if hasattr(self, 'save_dir') and self.save_dir:
                        try:
                            with open(f"{self.save_dir}/benchmark-1batch-1thread.json", "w", encoding="utf-8") as f:
                                json.dump(self.bench_mark_data, f, indent=2)
                            console.print(f"[green]Benchmark data saved to {self.save_dir}[/green]")
                        except Exception as e:
                            console.print(f"[red]Error saving benchmark data: {e}[/red]")
                    else:
                        console.print("[yellow]Warning: No save directory available for benchmark data[/yellow]")
                    
                    console.print(f"[bold magenta]End processing video: {session_id}[/bold magenta]")
                    
                    if hasattr(self, 'start_time'):
                        console.print(f"Time elapsed: {time.perf_counter() - self.start_time}")
                    
                    console.print(info.get("summary"))
                                    
            except Exception as e:
                console.print(f"\n[bold red]CRITICAL ERROR:[/bold red]")
                console.print(f"Error type: {type(e)}")
                console.print(f"Error message: {str(e)}")
                console.print(f"Traceback: {traceback.format_exc()}")
                
                # Emergency cleanup
                if 'video_writer' in locals():
                    try:
                        video_writer.release()
                    except:
                        pass
                if 'detection_writer' in locals():
                    try:
                        detection_writer.release()
                    except:
                        pass
                if hasattr(self, 'tracking_file') and self.tracking_file:
                    try:
                        self.tracking_file.close()
                    except:
                        pass
                
                # Try to continue processing
                continue