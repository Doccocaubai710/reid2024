from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import tqdm
from rich.console import Console

from . import constants as const
from .f1_og import feature_extractor
from .schemas import PersonID

console = Console()


def extract_embedding(
    images: List[np.ndarray],
    async_mode: bool = False,
) -> np.ndarray:
    if not async_mode:
        return feature_extractor.inference(images)

    return feature_extractor.inference_async(images)


def extract_embeddings_for_persons(image: np.ndarray, faces: list, bodys: list):
    """
    Extract embeddings from the image and detections

    Args:
        image: The image frame
        faces: List of face bounding boxes
        bodys: List of body bounding boxes

    Returns:
        List of PersonID objects
    """
    current_persons: List[PersonID] = []

    for body in bodys:
        if body.get("frame").tolist() == []:
            continue

        # Prepare input for BYTETrack (bbox, confidence, body_embedding)
        bbox = body.get("bbox")
        face = body.get("face", [])

        full_body_embedding = extract_embedding(
            [image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]],
            async_mode=const.ASYNC_MODE,
        )

        # if face != []:
        #     face_embed = extract_embedding(
        #         [image[face[1] : face[1] + face[3], face[0] : face[0] + face[2]]],
        #         async_mode=const.ASYNC_MODE,
        #     )
        # else:
        face_embed = None

        confidence = body.get("score")
        face_confidence = body.get("face_conf", [])

        person = PersonID(
            fullbody_embedding=full_body_embedding,
            face_embedding=face_embed,
            fullbody_bbox=bbox,
            face_bbox=face,
            body_conf=confidence,
            face_conf=face_confidence,
        )
        current_persons.append(person)

    return current_persons


def get_face_body_from_detection(image: np.ndarray, detections: list):
    """
    Process the detections to get body and face bounding boxes.
    """
    faces, bodys = [], []

    for detection in detections:
        x, y, w, h = list(map(lambda x: int(x), detection.get("bbox")))
        detection.update({"frame": image[y : y + h, x : x + w]})
        face_bbox = detection.get("face", [])
        bodys.append(detection)
        if face_bbox:
            faces.append(face_bbox)

    return faces, bodys


def check_inside(box_a: list, box_b: list) -> bool:
    """
    Check if box_a is inside box_b

    Args:
        box_a (list): [x1, y1, x2, y2]
        box_b (list): [x1, y1, x2, y2]

    The Origin is at the top-left corner,
    x-axis is the horizontal axis and y-axis is the vertical axis
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    if x1_a >= x1_b and y1_a >= y1_b and x2_a <= x2_b and y2_a <= y2_b:
        return True

    return False


class BatchGetEmbeddingsExecutor:
    def __init__(self, max_batch_size: int, max_thread: int):
        self.max_batch_size = max_batch_size
        self.max_thread = max_thread

        self.queue = []
        self.frame_info = []  # Mapping from frame in queue to metadata

    def add(
        self,
        image,
        detections: list,
        metadata: dict,
        session_id: int,
        is_skipped=False,
    ):
        self.queue.append(image)
        self.frame_info.append(
            {
                "detections": detections,
                "metadata": metadata,
                "session_id": session_id,
                "is_skipped": is_skipped,
            }
        )

    @property
    def queue_is_full(self):
        return len(self.queue) >= self.max_batch_size

    def process(self):
        output = []

        with ThreadPoolExecutor(max_workers=self.max_thread) as executor:
            futures = {}

            for i, image in enumerate(self.queue):
                detections = self.frame_info[i].get("detections")
                metadata = self.frame_info[i].get("metadata")
                session_id = self.frame_info[i].get("session_id")
                is_skipped = self.frame_info[i].get("is_skipped")

                if not is_skipped:
                    faces, bodys = get_face_body_from_detection(
                        image=image, detections=detections
                    )

                    future = executor.submit(
                        extract_embeddings_for_persons, image, faces, bodys
                    )

                    futures[future] = {
                        "idx": i,
                        "original_image": image,
                        "detections": detections,
                        "metadata": metadata,
                        "session_id": session_id,
                        "is_skipped": is_skipped,
                    }
                else:
                    console.print("[bold red]Skipped[/bold red] frame")
                    output.append(
                        {
                            "idx": i,
                            "original_image": image,
                            "detections": detections,
                            "metadata": metadata,
                            "session_id": session_id,
                            "is_skipped": is_skipped,
                            "persons": [],
                        },
                    )

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), unit="frames"
            ):
                info = futures[future]
                info["persons"] = future.result()

                output.append(info)

        # Sort the output by the original order
        output = sorted(output, key=lambda x: x.get("idx"))

        # Reset the queue
        self.queue = []
        self.frame_info = []

        return output
