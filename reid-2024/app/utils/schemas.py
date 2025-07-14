import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from app.utils import const

# Cấu hình logging
logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


class PersonID:
    def __init__(
        self,
        fullbody_embedding: np.ndarray,
        #face_embedding: Union[np.ndarray, None],
        fullbody_bbox: np.ndarray,
        #face_bbox: Union[np.ndarray, None],
        body_conf: np.float32,
        #face_conf: Union[np.float32, None],
    ):
        self.fullbody_embedding = fullbody_embedding
        #self.face_embedding = face_embedding
        self.fullbody_bbox = fullbody_bbox
        #self.face_bbox = face_bbox
        self.body_conf = body_conf
        #self.face_conf = face_conf
        self.ttl = const.TIME_TO_LIVE  # Frames to live before removing from storage
        self.smooth_body = None
        self.smooth_face = None
        self.fullbody_embeddings = None
        #self.face_features = []

        #self.face_score = []
        self.count = 0
        self.id = None

    def set_id(self, id: int):
        self.id = id

    # def add_face_embeddings(self, face_feature, face_score):
    #     face_feature /= np.linalg.norm(face_feature)
    #     if len(self.face_score) < 10:
    #         self.face_features.append(face_feature)
    #         self.face_score.append(face_score)
    #     elif face_score > min(self.face_score):
    #         position = self.face_score.index(min(self.face_score))
    #         self.face_score[position] = face_score
    #         self.face_features[position] = face_feature

    def add_fullbody_embeddings(self, body_feature, body_score):
        body_feature /= np.linalg.norm(body_feature)
        if self.fullbody_embeddings is None:
            self.fullbody_embeddings = body_feature
        else:
            self.count += 1
            self.fullbody_embeddings = (
                body_feature + self.fullbody_embeddings * (self.count - 1)
            ) / self.count

    def __str__(self) -> str:
        return (
            f"Person ID: {self.id}, TTL: {self.ttl}, Fullbody BBox: {self.fullbody_bbox}, Face BBox: {self.face_bbox}"
            f", Len Fullbody Embedding: {len(self.fullbody_embedding) if self.fullbody_embedding is not None else None}"
        )


class PersonIDsStorage:
    def __init__(self):
        self.person_ids = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        return np.dot(a, b) / (norm(a) * norm(b))

    def get_person_by_id(self, id):
        for person in self.person_ids:
            if person.id == id:
                return person

    def search(
        self,
        current_person_id: PersonID,
        current_frame_id: List[PersonID],
        threshold: float = 0.2,
    ) -> Union[PersonID, None]:
        """
        Loop through all existing Person Ids and find the closest one
        """
        most_match = None
        min_similarity = 1.0

        # Normalize embeddings
        current_body_emb = F.normalize(
            torch.tensor(current_person_id.fullbody_embedding), dim=0
        ).numpy()

        # current_face_emb = (
        #     F.normalize(torch.tensor(current_person_id.face_embedding), dim=0).numpy()
        #     if current_person_id.face_embedding is not None
        #     else None
        # )

        for person_id in self.person_ids:
            if person_id.id not in current_frame_id:
                # Normalize stored embeddings
                # logging.info(f"Len embeddings: {len(person_id.fullbody_embeddings)}")

                # if person_id.face_embedding is None:
                #     continue
                if person_id.fullbody_embeddings is None:
                    continue
                if len(person_id.fullbody_embeddings) > 0:
                    person_body = F.normalize(
                        torch.tensor(person_id.fullbody_embeddings), dim=0
                    ).numpy()
                    similarity = np.maximum(
                        0.0,
                        cdist(
                            current_body_emb.reshape(1, -1),
                            person_body.reshape(1, -1),
                            metric="cosine",
                        ),
                    )

                    logging.info(f"Similarity: {similarity}")

                    if similarity < min_similarity and similarity < threshold:
                        most_match = person_id
                        min_similarity = similarity
                        logging.info(f"{min_similarity}, {most_match.id}")

        return most_match, min_similarity

    def add(self, person_id: PersonID):
        self.person_ids.append(person_id)


