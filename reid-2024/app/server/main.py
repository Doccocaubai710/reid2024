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
from app.deeputils.bytetracker import BYTETracker
from app.utils import const
from app.utils.benchmark import BenchmarkMultiThread
from app.utils.embeddings import extract_embedding
from app.utils.schemas import PersonID, PersonIDsStorage

console = Console()
logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

def extract_embeddings(image: np.ndarray, bodys: list):
    """Extract embeddings from the image and detections"""
    current_persons: List[PersonID] = []
    
    for body in bodys:
        if body.get("frame").tolist() == []:
            continue
            
        bbox = body.get("bbox")
        full_body_embedding = extract_embedding(
            [image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]],
            async_mode=const.ASYNC_MODE,
        )
        
        confidence = body.get("score")
        person = PersonID(
            fullbody_embedding=full_body_embedding,
            fullbody_bbox=bbox,
            body_conf=confidence,
        )
        current_persons.append(person)
    
    return current_persons

def get_face_body_from_detection(image: np.ndarray, detections: list):
    """Process the detections to get body and face bounding boxes."""
    faces, bodys = [], []
    
    for detection in detections:
        x, y, w, h = list(map(lambda x: int(x), detection.get("bbox")))
        detection.update({"frame": image[y : y + h, x : x + w]})
        bodys.append(detection)
        
    
    return bodys


class MainServer:
    def __init__(self):
        self.receiver = None
        self.video_writers = {}
        self.storage = PersonIDsStorage()
        self.tracking_file = None
        self.bench_mark_data = {}
        self.log_step = [10, 40, 100, 200, 400]
        self.process = psutil.Process()
        
        
        
    def check_system_limits(self):
        """Check various system limits"""
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            #console.print(f"File descriptor limits: soft={soft_limit}, hard={hard_limit}")
            
            current_fds = self.process.num_fds()
            #console.print(f"Current file descriptors: {current_fds}")
            
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
        #console.print(f"Disk free: {disk_usage.free//1e9}GB")
        
        # Check video writer status
        if session_id in sessions_storage:
            video_writer = sessions_storage[session_id]
            console.print(f"VideoWriter exists: {video_writer is not None}")
            if video_writer:
                # Check if video writer is still open
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
            
            # Bind to all interfaces on port 5556
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
                    msg = packet["meta"]  # msg vẫn đang là JSON string
                    
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
    
    # def init_receiver(self) -> bool:
    #     try:
    #         import zmq
    #         import imagezmq
            
    #         # Initialize the receiver first using the standard approach
    #         self.receiver = imagezmq.ImageHub(
    #             open_port="tcp://0.0.0.0:5555", 
    #             REQ_REP=False
    #         )
            
            
    #         # ZMQ_RCVHWM (High Water Mark for receiving) controls message buffering
    #         self.receiver.zmq_socket.setsockopt(zmq.RCVHWM, 10000)  # Increase from default 1000 to 10000
            
            
    #         # ZMQ_RCVBUF - Kernel receive buffer size
    #         self.receiver.zmq_socket.setsockopt(zmq.RCVBUF, 32 * 1024 * 1024)  # 32MB buffer
            
    #         # ZMQ_LINGER - Socket linger period for unsent messages
    #         self.receiver.zmq_socket.setsockopt(zmq.LINGER, 1000)  # 1 second
            
    #         console.print("[bold cyan]Image Receiver[/bold cyan] initialized [bold green]successfully[/bold green] with increased buffer capacity")
    #         return True
            
    #     except Exception as e:
    #         console.print(f"[bold red]Error when initializing Image Receiver: {str(e)}[/bold red]")
    #         # Print detailed traceback for debugging
    #         import traceback
    #         console.print(f"[red]{traceback.format_exc()}[/red]")
    #         return False

    def draw_detection(self, img, bboxes, scores, ids, mask_alpha=0.3):
        height, width = img.shape[:2]
        np.random.seed(0)
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(1, 3))
        mask_img = img.copy()
        det_img = img.copy()
        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)
        for bbox, score, id_ in zip(bboxes, scores, ids):
            color = colors[0]
            bbox = np.array(bbox)
            x1, y1, x2, y2 = bbox.astype(np.int64)
            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            caption = f"body {int(score*100)}% ID: {id_}"
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=size,
                thickness=text_thickness,
            )

            th = int(th * 1.2)
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(
                det_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
    def draw_detection_only(self, img, bboxes, scores, mask_alpha=0.3):
        """
        Draw detection boxes with confidence scores only (no tracking IDs)
        Boxes are in format [x, y, w, h] where (x,y) is the top-left corner
        """
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
            # Extract coordinates directly in x,y,w,h format
            x, y, w, h = bbox.astype(np.int64)
            
            # Draw rectangle using top-left point (x,y) and width/height (w,h)
            cv2.rectangle(det_img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(mask_img, (x, y), (x + w, y + h), color, -1)
            
            # Only include confidence score, not ID
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

    def reset_count(self, id):
        BaseTrack._count = id
    

    def remap_bytetracker_ids(self, bytetracker, old_id, new_id):
        """Remap a track ID in BYTETracker from old_id to new_id."""
        for track in bytetracker.tracked_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(f"Remapped BYTETracker ID from {old_id} to {new_id}")
                break

        for track in bytetracker.lost_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(f"Remapped BYTETracker ID (lost track) from {old_id} to {new_id}")
                break

        for track in bytetracker.removed_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(
                    f"Remapped BYTETracker ID (removed track) from {old_id} to {new_id}"
                )
                break

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
            # And change file extension
            save_path = f"{save_dir}/{os.path.basename(metadata.get('source'))}.avi"
             # Create features directory
            features_dir = f"{save_dir}/features"
            os.makedirs(features_dir, exist_ok=True)
            console.print(f"[yellow]Saving video to: {os.path.abspath(save_path)}[/yellow]")
            
            # Check if file can be created
            try:
                self.tracking_file = open(f"{save_dir}/tracking_results.txt", "w")
                console.print("[green]Tracking file opened successfully[/green]")
                #Intialize feature storage
                self.feature_storage = {}
                self.feature_storage_file = f"{features_dir}/feature_vectors.pkl"
                console.print(f"[green]Feature storage initialized at {self.feature_storage_file}")
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
    def store_feature_vectors(self, frame_number, tracking_ids, bboxes, features, confidences):
        """
        Store feature vectors for a frame with multiple people, appending to a single file
        after processing each frame.
        
        Args:
            frame_number: Current frame number
            tracking_ids: Array of tracking IDs
            bboxes: Array of bounding boxes
            features: Array of feature vectors
            confidences: Array of confidence scores
        """
        try:
            # Ensure features directory exists
            if hasattr(self, 'save_dir'):
                features_dir = f"{self.save_dir}/features"
                os.makedirs(features_dir, exist_ok=True)
                
                # Determine the feature file path
                feature_file = f"{features_dir}/all_features.jsonl"
                
                # Process and prepare the frame data
                frame_data = {
                    "frame": int(frame_number),
                    "detections": []
                }
                
                for i in range(len(tracking_ids)):
                    if i < len(bboxes) and i < len(features) and i < len(confidences):
                        # Process bbox format
                        bbox = bboxes[i]
                        if len(bbox) == 4:
                            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # It's [x1, y1, x2, y2]
                                x1, y1, x2, y2 = bbox
                                w, h = x2 - x1, y2 - y1
                                x, y = x1, y1
                            else:  # It's already [x, y, w, h]
                                x, y, w, h = bbox
                        else:
                            console.print(f"[red]Unexpected bbox format: {bbox}[/red]")
                            continue
                        
                        # Convert feature to numpy if it's a tensor
                        if hasattr(features[i], 'cpu'):
                            feature_np = features[i].cpu().detach().numpy().tolist()  # Convert to list for JSON
                        else:
                            feature_np = features[i].tolist() if hasattr(features[i], 'tolist') else features[i]
                        
                        # Create person data
                        person_data = {
                            "track_id": int(tracking_ids[i]),
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "feature": feature_np,
                            "confidence": float(confidences[i])
                        }
                        
                        frame_data["detections"].append(person_data)
                
                # Append frame data to the file
                with open(feature_file, 'a') as f:
                    f.write(json.dumps(frame_data) + '\n')
                    
                console.print(f"[green]Saved {len(frame_data['detections'])} features for frame {frame_number}[/green]")
            else:
                console.print("[yellow]Cannot save features: save_dir not set[/yellow]")
        except Exception as e:
            console.print(f"[red]Error storing feature vectors for frame {frame_number}: {e}[/red]")
            console.print(traceback.format_exc())
    def save_feature_storage(self):
        """Save the feature storage to disk"""
        try:
            if hasattr(self, 'feature_storage') and hasattr(self, 'feature_storage_file'):
                # Make sure the directory exists
                os.makedirs(os.path.dirname(self.feature_storage_file), exist_ok=True)
                
                # Count total detections
                total_detections = sum(len(people) for people in self.feature_storage.values())
                with open(self.feature_storage_file, 'wb') as f:
                    pickle.dump(self.feature_storage, f)
                    
                console.print(f"[green]Feature storage saved with {total_detections} detections across {len(self.feature_storage)} frames[/green]")
        except Exception as e:
            console.print(f"[red]Error saving feature storage: {e}[/red]")
            console.print(traceback.format_exc())     
                
            
    def run(self):
        ############################## INITIALIZATION ##############################
        if not self.init_receiver():
            console.print("[bold red] Error when initializing receiver [/bold red]")
            return
            
        frame_count = 0
        bytetrack = BYTETracker()
        tracked_ids = np.array([], dtype=np.int32)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI format

        sessions_storage = {}  # Store the video writer for each session (source)
        # Add storage for detection-only video writers
        self.detection_video_writers = {}
        
        # Initial system check
        console.print("\n[bold cyan]Initial System Check[/bold cyan]")
        self.check_system_limits()
        self.check_memory_usage()
        
        while True:
            try:
                #Monitor resources before receiving
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
                    
                    # Check if step_ids exists and increment
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
                    # Check video writers before use
                    if video_writer is None:
                        console.print(f"[red]ERROR: No tracking video writer for session {session_id}[/red]")
                        continue
                    
                    if detection_writer is None:
                        console.print(f"[red]ERROR: No detection video writer for session {session_id}[/red]")
                        continue
                    
                    # Process detections
                    bodys = get_face_body_from_detection(opencv_image, detections)
                    _time = time.perf_counter()
                    current_persons = extract_embeddings(opencv_image, bodys)
                    console.print(f"Extract embeddings time: {time.perf_counter() - _time}")

                    # Extract necessary data for tracking
                    boxes = np.asarray([current_person.fullbody_bbox for current_person in current_persons])
                    confidences = np.asarray([current_person.body_conf for current_person in current_persons])
                    # plr_osnet here
                    track_bodys = np.asarray([current_person.fullbody_embedding for current_person in current_persons])
                    # track_bodys_list = []
                    # for current_person in current_persons:
                    #     feature = current_person.fullbody_embedding
                        
                    #     # Handle different feature formats
                    #     if isinstance(feature, torch.Tensor):
                    #         feature = feature.cpu().detach().numpy()
                        
                    #     # Ensure feature is 1D (flattened)
                    #     if feature.ndim > 1:
                    #         feature = feature.flatten()
                        
                    #     track_bodys_list.append(feature)

                    # track_bodys = np.array(track_bodys_list)        
                    # Create detection-only visualization (before tracking)
                    detection_img = opencv_image.copy()
                    if len(boxes) > 0:
                        # Create detection-only visualization
                        detection_img = self.draw_detection_only(
                            img=opencv_image.copy(), 
                            bboxes=boxes, 
                            scores=confidences
                        )
                    
                    bboxes, scores, ids = bytetrack.update(boxes, confidences, track_bodys)
                    tracking_ids = np.array(ids).astype(np.int32)
                    new_ids = np.setdiff1d(tracking_ids, tracked_ids)
                    if new_ids.size > 0:
                        logging.info(f"{tracking_ids}")
                        logging.info(f"{new_ids}")

                    # Update the storage
                    actual_new_ids = []

                    _time = time.perf_counter()
                    # Process tracking results
                    if len(bboxes) > 0:
                        persons = [current_persons[i] for i in range(len(current_persons)) if confidences[i] in scores]
                        
                        # Debug: print number of persons and their confidence scores
                        console.print(f"Number of persons detected: {len(persons)}")
                        console.print(f"Confidence scores: {[p.body_conf for p in persons]}")
                        
                        # Always store features for all tracked persons
                        features_to_store = []
                        
                        for i, person in enumerate(persons):
                            
                            person.set_id(tracking_ids[i])
                            track_id = tracking_ids[i]
                            
                            # Always store the feature vector for this person
                            features_to_store.append(person.fullbody_embedding)
                            
                            # Rest of your existing person processing code...
                            if frame_number == 0:
                                person.set_id(tracking_ids[i])
                                self.storage.add(person)
                                actual_new_ids.append(tracking_ids[i])
                                if person.body_conf > 0.6:
                                    person.add_fullbody_embeddings(
                                        person.fullbody_embedding, person.body_conf
                                    )
                            elif track_id not in new_ids:
                                old_person = self.storage.get_person_by_id(track_id)
                                if person.body_conf > 0.6:
                                    old_person.add_fullbody_embeddings(
                                        person.fullbody_embedding, person.body_conf
                                    )
                            else: 
                                actual_new_ids.append(track_id)
                                person.set_id(track_id)
                                self.storage.add(person)
                                if person.body_conf > 0.6:
                                    person.add_fullbody_embeddings(
                                        person.fullbody_embedding, person.body_conf
                                    )
                            tracking_ids[i] = person.id

                        
                        # Store all features regardless of confidence
                        try:
                            if len(features_to_store) > 0:  # Only attempt if we have at least one feature
                                self.store_feature_vectors(
                                    frame_number=frame_number,
                                    tracking_ids=tracking_ids,  # Ensure matching sizes
                                    bboxes=bboxes,
                                    features=features_to_store,
                                    confidences=scores
                                )
                                console.print(f"[green]Successfully stored {len(features_to_store)} features for frame {frame_number}[/green]")
                            else:
                                console.print("[yellow]No features to store for this frame[/yellow]")
                        except Exception as e:
                            console.print(f"[red]Error storing features: {e}[/red]")
                            console.print(traceback.format_exc())  # Print full traceback
                                
                        logging.info(f"Tracking id: {tracking_ids}")
                        tracked_ids = np.concatenate((tracked_ids, actual_new_ids))
                        logging.info(f"Tracked_id: {tracked_ids}")
                        if len(tracked_ids) > 0:  # Check if the array is not empty
                            self.reset_count(int(np.max(tracked_ids)))
                        else:
                            # Handle the case when tracked_ids is empty
                            # You might want to set a default value or skip this step
                            console.print("[yellow]No tracked IDs to reset count[/yellow]")
                        conf_scores = np.array(scores).astype(np.float64)
                        boxes = np.array(bboxes).astype(np.float64)
                        
                        # Write tracking data to file
                        try:
                            for i in range(len(bboxes)):
                                line = f"{frame_number},{tracking_ids[i]},{bboxes[i][0]},{bboxes[i][1]},{bboxes[i][2]},{bboxes[i][3]},{scores[i]}\n"
                                self.tracking_file.write(line)
                                self.tracking_file.flush()
                        except Exception as e:
                            console.print(f"[red]Error writing to tracking file: {e}[/red]")
                        
                        # Draw tracking results
                        result_img = self.draw_detection(img=opencv_image, bboxes=bboxes, scores=scores, ids=tracking_ids)
                        
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
                    
                    # Save the benchmark data only if save_dir exists
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