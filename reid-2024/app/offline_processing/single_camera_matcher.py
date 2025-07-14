import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import logging
import argparse
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import copy
from sklearn.metrics import silhouette_score
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("single_camera_matching.log"),
        logging.StreamHandler()
    ]
)

class SingleCameraMatcher:
    """ 
    Implementation of single-camera matching to group tracklets
    """
    def __init__(self,camera_id,outzone_file=None):
        self.camera_id=camera_id
        self.tracklets={}
        self.clusters={}
        self.uncertain_tracklets=[]
        # Load outzone information if provided
        self.outzones = []
        if outzone_file and os.path.exists(outzone_file):
            try:
                with open(outzone_file, 'r') as f:
                    self.outzones = json.load(f)
                logging.info(f"Loaded outzones for camera {camera_id}")
            except Exception as e:
                logging.error(f"Failed to load outzones: {e}")
        
    def add_tracklet(self,tracklet):
        self.tracklets[tracklet.track_id]=tracklet
    def load_tracklets(self,tracklets_file):
        """Load tracklets from a pickle file"""
        try:
            with open(tracklets_file, 'rb') as f:
                tracklets = pickle.load(f)
                
            for tracklet in tracklets:
                self.add_tracklet(tracklet)
                
            logging.info(f"Loaded {len(tracklets)} tracklets for camera {self.camera_id}")
        except Exception as e:
            logging.error(f"Failed to load tracklets: {e}")
    def filer_reliable_tracklets(self,min_length=10,outzone_threshold=0.8):
        """
        Filter tracklets to separate reliable and unreliable ones
        
        Args:
            min_length: Minimum number of frames for a tracklet to be considered reliable
            outzone_threshold: Maximum proportion of frames in an outzone for a reliable tracklet
        
        Returns:
            reliable_tracklets, unreliable_tracklets
        """
        reliable = []
        unreliable = []
        
        for tracklet_id, tracklet in self.tracklets.items():
            # Check tracklet length
            if len(tracklet.frames) < min_length:
                unreliable.append(tracklet)
                continue
            
            # Check if tracklet is mostly in outzone
            if self.outzones:
                outzone_frames = 0
                for bbox in tracklet.bboxes:
                    if self.is_in_outzone(bbox):
                        outzone_frames += 1
                
                outzone_ratio = outzone_frames / len(tracklet.frames)
                if outzone_ratio > outzone_threshold:
                    unreliable.append(tracklet)
                    continue
            
            reliable.append(tracklet)
        logging.info(f"Filtered {len(reliable)} reliable and {len(unreliable)} unreliable tracklets")
        return reliable, unreliable
    
    def is_in_outzone(self,bbox):
        if not self.outzones:
            return False
        x,y,w,h=bbox
        for zone in self.outzones:
            zone_x1, zone_y1, zone_x2, zone_y2 = zone
            
            # Check if bbox center is in zone
            center_x = x + w/2
            center_y = y + h/2
            
            if (zone_x1 <= center_x <= zone_x2) and (zone_y1 <= center_y <= zone_y2):
                return True
                
        return False
    def initialize_clusters(self,reliable_tracklets,method="anchor_frame",anchor_frame=None):
        """
        Initialize clusters from reliable tracklets
        
        Args:
            reliable_tracklets: List of reliable tracklets
            method: Method to initialize clusters ("anchor_frame" or "silhouette")
            anchor_frame: Specific frame to use as anchor (if method is "anchor_frame")
        """
        clusters={}
        if method=="anchor_frame" and anchor_frame is not None:
            cluster_idx=0
            for tracklet in reliable_tracklets:
                if anchor_frame in tracklet.frames:
                    clusters[cluster_idx]=[tracklet.track_id]
                    cluster_idx+=1
            logging.info(f"Initialized {len(clusters)} clusters from anchor frame {anchor_frame}")
        
        elif method=="silhouette":
            if len(reliable_tracklets)<2:
                logging.warning("Not enough reliable tracklets for clustering")
                return clusters
            features = np.array([t.mean_features() for t in reliable_tracklets])
            tracklet_ids = [t.track_id for t in reliable_tracklets]
            
            distance_matrix=cdist(features,features,'cosine')
            
            # Check time overlap
            for i in range(len(reliable_tracklets)):
                for j in range(i+1,len(reliable_tracklets)):
                    if self.check_time_overlap(reliable_tracklets[i],reliable_tracklets[j]):
                        distance_matrix[i,j]=1.0
                        distance_matrix[j,i]=1.0

            best_score=-1
            best_n_clusters=1
            
            #Try different numbers of clusters
            for n_clusters in range(2,min(10,len(reliable_tracklets))):
                clustering=AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    linkage="average"
                )
                labels=clustering.fit_predict(distance_matrix)
                if len(set(labels))>1:
                    score=silhouette_score(distance_matrix,labels,metric="precomputed")
                    if score>best_score:
                        best_score=score
                        best_n_clusters=n_clusters
            
            #Perform final clustering
            clustering=AgglomerativeClustering(
                n_clusters=best_n_clusters,
                affinity='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
            
            #Creaate clusters
            for i,label in enumerate(labels):
                if label not in clusters:
                    clusters[label]=[]
                clusters[label].append(tracklet_ids[i])
        
        self.clusters=clusters
        return clusters 
    
    def r_matching(self,reliable_tracklets,unreliable_tracklets,similarity_threshold=0.3,min_err_threshold=0.1):
        """Implement of R-matching algorithm

        Args:
            reliable_tracklets (_type_): _description_
            unreliable_tracklets (_type_): _description_
            similarity_threshold (float, optional): _description_. Defaults to 0.3.
            min_err_threshold (float, optional): _description_. Defaults to 0.1.
        """
        #Make sure to have clusters initialized
        if not self.clusters:
            logging.warning("No clusters initialized")
            return
        
        #Create mapping from tracklet ID to tracklet
        tracklets_by_id={t.track_id: t for t in reliable_tracklets+unreliable_tracklets}
        
        #Compute cluster features
        cluster_features={}
        for cluster_id,tracklet_ids in self.clusters.items():
            features=[]
            for tid in tracklet_ids:
                if tid in tracklets_by_id:
                    features.append(tracklets_by_id[tid].mean_features())
            
            if features:
                cluster_features[cluster_id]=np.mean(np.array(features),axis=0)
                
        cos=nn.CosineSimilarity(dim=0)
        uncertain_tracklets=[]
        for tracklet in unreliable_tracklets:
            candidates=[]
            for cluster_id,cluster_feature in cluster_features.items():
                # Skip if time periods overlap
                if any(self.check_time_overlap(tracklet, tracklets_by_id[tid]) 
                       for tid in self.clusters[cluster_id] if tid in tracklets_by_id):
                    continue
                #Calculate sim
                tracklet_feature=torch.tensor(tracklet.mean_features(),dtype=torch.float32)
                cluster_feature_tensor=torch.tensor(cluster_feature,dtype=torch.float32)
                cosine_sim=cos(tracklet_feature,cluster_feature_tensor).item()
                distance=1.0-cosine_sim
                candidates.append((distance,cluster_id))  
            
            if not candidates:
                uncertain_tracklets.append(tracklet)
                continue
            candidates.sort(key=lambda x:x[0])
            # If best match is too distant, mark as uncertain
            if candidates[0][0]>similarity_threshold:
                uncertain_tracklets.append(tracklet)
                continue
            # Otherwise, assign to best matching cluster
            best_cluster=candidates[0][1]
            self.clusters[best_cluster].append(tracklet.track_id)
            
            #Update cluster feature
            tracklet_ids=self.cluster[best_cluster]
            features=[tracklets_by_id[tid].mean_features() for tid in tracklet_ids if tid in tracklets_by_id]
            cluster_features[best_cluster]=np.mean(np.array(features),axis=0)
            
        self.uncertain_tracklets=uncertain_tracklets
        logging.info(f"R-matching completed. {len(uncertain_tracklets)} tracklets remain uncertain.")
        return self.clusters,uncertain_tracklets
    
    def check_time_overlap(self,tracklet1,tracklet2):
        frames1=set(tracklet1.frames)
        frames2=set(tracklet2.frames)
        return len(frames1.intersection(frames2)>0)
    
    def match_all_tracklets(self,min_length=10,outzone_threshold=0.8,
                            method="anchor_frame",anchor_frame=None,
                            similarity_threshold=0.3,min_err_threshold=0.1):
        """Complete process to match all tracklets
        
        Args:
            min_length: Minimum number of frames for a tracklet to be considered reliable
            outzone_threshold: Maximum proportion of frames in an outzone for a reliable tracklet
            method: Method to initialize cluster( "anchor frame" or silhoulette)
            anchor_frame: Specific frame to use as anchor
            similarity_threshold: Maximum cosine distance for a match
            min_err_threshold: minimum difference between best and second best match"""
            
        
        #Step 1: Filter tracklets into reliable and unreliable 
        reliable,unreliable=self.filer_reliable_tracklets(min_length,outzone_threshold)
        # If no reliable tracklets, we can't proceed
        if not reliable:
            logging.warning("No reliable tracklets found")
            return {}
        #Step 2: Initialize clusters from reliable tracklets
        self.initialize_clusters(reliable,method,anchor_frame)
        
        #Step 3: Match unreliable tracklets using R-matching
        clusters,uncertain=self.r_matching(
            reliable,unreliable,similarity_threshold,min_err_threshold
        )
        return clusters
    
    def save_results(self,output_file,clusters_file=None):
        """
        Save the results
        """
        tracklet_to_cluster={}
        for cluster_id,tracklet_ids in self.clusters.items():
            for tid in tracklet_ids:
                tracklet_to_cluster[tid]=cluster_id
        
        with open(output_file,'w') as f:
            for tracklet_id,tracklet in self.tracklets.items():
                cluster_id=tracklet_to_cluster.get(tracklet_id,-1) #-1 for uncertain tracklet
                
                for i in range(len(tracklet.frames)):
                    frame=tracklet.frames[i]
                    bbox=tracklet.bboxes[i]
                    #Format camera_id, cluster_id,frame,x,y,w,h,-1,-1
                    line = f"{self.camera_id},{cluster_id},{frame},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},-1,-1\n"
                    f.write(line)
        # Save clusters if requested
        if clusters_file:
            with open(clusters_file, 'w') as f:
                json.dump(self.clusters, f, indent=4)
        
        logging.info(f"Saved matching results to {output_file}")
        if clusters_file:
            logging.info(f"Saved clusters to {clusters_file}")
    
def process_camera(tracklets_file,output_file,clusters_file=None,outzone_file=None,anchor_frame=None):
    """Process single-camera matching for one camera"""
    camera_id=os.path.basename(tracklets_file).split('_')[0]
    matcher=SingleCameraMatcher(camera_id,outzone_file)
    matcher.load_tracklets(tracklets_file)
    #Perform matching
    if anchor_frame is None:
        anchor_frames = {
            'c001': 6281,
            'c002': 19568,
            'c003': 17233,
            'c005': 17286,
            'c006': 40007,
            'c007': 32904
        }
        anchor_frame=anchor_frames.get(camera_id)
    matcher.match_all_tracklets(
        method="anchor_frame" if anchor_frame else 'silhouette',
        anchor_frame=anchor_frame
    )
    matcher.save_results(output_file,clusters_file)

def process_directory(input_dir,output_dir,clusters_dir=None,outzone_dir=None):
    """Process single-camera matching for all cameras"""
    
    os.makedirs(output_dir,exist_ok=True)
    if clusters_dir:
        os.makedirs(clusters_dir,exist_ok=True)
    processed_files=0
    for filename in os.listdir(input_dir):
        if filename.endswith('_tracklets.pkl'):
            tracklets_file=os.path.join(input_dir,filename)
            camera_id=filename.split('_')[0]
            output_file=os.path.join(output_dir,f"{camera_id}_matched.txt")
            clusters_file = None
            if clusters_dir:
                clusters_file = os.path.join(clusters_dir, f"{camera_id}_clusters.json")
            
            outzone_file = None
            if outzone_dir:
                outzone_file = os.path.join(outzone_dir, f"{camera_id}_outzones.json")
                if not os.path.exists(outzone_file):
                    outzone_file = None
            
            process_camera(tracklets_file, output_file, clusters_file, outzone_file)
            processed_files += 1
    logging.info(f"Processed {processed_files} cameras")
                
                        