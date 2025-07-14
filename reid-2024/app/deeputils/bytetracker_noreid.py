import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from app.deeputils.kalman_filter import KalmanFilter
from app.deeputils import matching
from app.deeputils.base_track import BaseTrack,TrackState
# REMOVED: from app.utils.schemas import PersonID,PersonIDsStorage
import logging

logging.basicConfig(filename='example.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

class STrack(BaseTrack):
    shared_kalman=KalmanFilter()
    def __init__(self,tlwh,score,buffer_size=30):
        """
        Simplified STrack constructor - NO FEATURES/EMBEDDINGS
        Only uses bounding box (tlwh) and confidence score
        """
        self._tlwh=np.asarray(tlwh,dtype=np.float64)
        self.kalman_filter=None
        self.mean,self.covariance=None,None 
        self.is_activated=False 
        self.score=score
        self.tracklet_len=0
        # REMOVED all feature/embedding related attributes
        
    # REMOVED: update_bodys method - not needed without features

    def predict(self):
        mean_state=self.mean.copy()
        if self.state!=TrackState.Tracked:
            mean_state[7]=0
        self.mean,self.covariance=self.kalman_filter.predict(mean_state,self.covariance)
    @staticmethod
    def multi_predict(stracks):
        if len(stracks)>0:
            multi_mean=np.asarray([st.mean.copy() for st in stracks])
            multi_covariance=np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state!=TrackState.Tracked:
                    multi_mean[i][7]=0
            multi_mean,multi_covariance=STrack.shared_kalman.multi_predict(multi_mean,multi_covariance)
            for i,(mean,cov) in enumerate(zip(multi_mean,multi_covariance)):
                stracks[i].mean=mean
                stracks[i].covariance=cov
    def activate(self,kalman_filter,frame_id):
        """Start a new tracklet"""
        self.kalman_filter=kalman_filter
        self.track_id=self.next_id()
        self.mean,self.covariance=self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len=0
        self.state=TrackState.Tracked
        if frame_id==1:
            self.is_activated=True
        self.frame_id=frame_id
        self.start_frame=frame_id
    def re_activate(self,new_track,frame_id,new_id=False):
        """Re-activate a lost track - simplified without features"""
        self.mean,self.covariance=self.kalman_filter.update(self.mean,self.covariance,self.tlwh_to_xyah(new_track._tlwh))
        self._tlwh=new_track._tlwh
        self.tracklet_len =0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
    def update(self,new_track,frame_id):
        """Update a matched track - simplified without features"""
        self.frame_id=frame_id
        self.tracklet_len+=1
        new_tlwh=new_track._tlwh
        self._tlwh=new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @staticmethod
    def reset_id():
        STrack.count = 0
    @property
    def tlwh(self):
        #Convert x_center,y_center,a,h to x1,y1,w,h
        if self.mean is None:
            return self._tlwh.copy()
        ret=self.mean[:4].copy()
        ret[2]*=ret[3]
        ret[:2]-=ret[2:]/2
        return ret
    @property
    def tlbr(self):
        #Convert x1,y1,w,h to x1,y1,x2,y2
        ret=self.tlwh.copy()
        ret[2:]+=ret[:2]
        return ret
    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret=np.asarray(tlwh).copy()
        ret[:2]+=ret[2:]/2
        ret[2]/=ret[3]
        return ret
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class BYTETracker:
    def __init__(self,frame_rate=30):
        self.tracked_stracks=[] #tracked in latest frame
        self.lost_stracks=[] #lost in latest frame
        self.removed_stracks=[] #lost has been removed
        self.frame_id=0
        self.low_thresh=0.25 #Threshold value for matching
        self.track_thresh=0.6 #High thresh
        self.det_thresh=self.track_thresh+0.05
        self.buffer_size=int(frame_rate / 30.0 * 30)
        self.max_time_lost=self.buffer_size
        self.kalman_filter=KalmanFilter()
        self.output_stracks=[]
        # Add switch candidates storage
        self.switch_candidates = {}
        # Define frame edge thresholds (% of frame size)
        self.edge_thresh_x = 0.10  # 10% from left/right edges
        self.edge_thresh_y = 0.15  # 15% from top/bottom edges
        # Default frame size (will be updated based on detections)
        self.frame_width = 1920
        self.frame_height = 1080
        
    def is_near_edge(self, tlwh):
        """
        Check if a bounding box is near the edge of the frame
        
        Args:
            tlwh: Bounding box in format [x, y, w, h] (top-left x, y, width, height)
            
        Returns:
            Tuple of (is_near_edge, edge_type)
        """
        x, y, w, h = tlwh
        
        # Update frame dimensions if detections go beyond current estimates
        right_edge = x + w
        bottom_edge = y + h
        if right_edge > self.frame_width:
            self.frame_width = right_edge
        if bottom_edge > self.frame_height:
            self.frame_height = bottom_edge
            
        # Calculate center point
        center_x = x + w/2
        center_y = y + h/2
        
        # Calculate edge boundaries
        left_boundary = self.frame_width * self.edge_thresh_x
        right_boundary = self.frame_width * (1 - self.edge_thresh_x)
        top_boundary = self.frame_height * self.edge_thresh_y
        bottom_boundary = self.frame_height * (1 - self.edge_thresh_y)
        
        # Check each edge
        near_left = center_x < left_boundary
        near_right = center_x > right_boundary
        near_top = center_y < top_boundary
        near_bottom = center_y > bottom_boundary
        
        # Determine edge type
        edge_type = None
        if near_left:
            edge_type = "left"
        elif near_right:
            edge_type = "right"
        elif near_top:
            edge_type = "top"
        elif near_bottom:
            edge_type = "bottom"
        
        is_near = near_left or near_right or near_top or near_bottom
        
        return is_near, edge_type
        
    def update(self,bboxes,scores,track_bodys=None):
        """
        PURE IoU-BASED TRACKING - NO EMBEDDINGS
        track_bodys parameter is ignored (kept for compatibility)
        """
        self.frame_id+=1
        activated_stracks=[]
        refind_stracks=[]
        lost_stracks=[]
        removed_stracks=[]
        
        # COMPLETELY IGNORE track_bodys - we only use IoU matching
        
        remain_inds=scores>self.track_thresh #High confidence
        indices_low=scores>self.low_thresh
        indices_high=scores<self.track_thresh
        indices_second=np.logical_and(indices_low,indices_high) #Low confidence

        boxes_keep=bboxes[remain_inds]
        scores_keep=scores[remain_inds]
        
        boxes_second=bboxes[indices_second]
        scores_second=scores[indices_second]

        if len(boxes_keep)>0:
            # Create detections WITHOUT features - only bbox and score
            detections = [STrack(tlwh,s) for (tlwh,s) in zip(boxes_keep,scores_keep)]
        else:
            detections=[]
        """Step 1: Add newly detected tracklets to tracked_stracks"""
        unconfirmed=[]
        tracked_stracks=[]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        """Step2: First association - PURE IoU MATCHING (no embedding distance)"""
        strack_pool=joint_stracks(tracked_stracks,self.lost_stracks)
        for track in strack_pool:
            logging.info(f"State: {track}, {track.state}")
        
        STrack.multi_predict(strack_pool)
        
        # PURE IoU DISTANCE - no embedding distance at all
        dists=matching.iou_distance(strack_pool,detections)
        logging.info(f"Dist0 (IoU only): {dists}")
        matches,u_track,u_detection=matching.linear_assignment(dists,thresh=0.8)
        
        logging.info(f"{matches},{u_track},{u_detection}")
        
        for tracked_i,box_i in matches:
            track=strack_pool[tracked_i]
            box=detections[box_i]
            if track.state==TrackState.Tracked:
                track.update(box,self.frame_id)
                activated_stracks.append(track)   
            else:
                track.re_activate(box,self.frame_id,new_id=False)
                refind_stracks.append(track)
                                    
        """Step 3: Second association, with IOU"""
        detections=[detections[i] for i in u_detection]
        r_tracked_stracks=[strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked or strack_pool[i].state==TrackState.Lost]
        logging.info(f"Step2: detections: {detections},remaining_track:{r_tracked_stracks}")
        dists=matching.iou_distance(r_tracked_stracks,detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, r_tracked_stracks, detections)
        logging.info(f"dists2: {dists}")
        matches,u_track,u_detection=matching.linear_assignment(dists,thresh=0.8)
        
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
                
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                

        """Step 3.5: Second association with low confidence detections"""
        if len(boxes_second)>0:
            # Create second detections WITHOUT features
            detections_second = [STrack(tlwh,s) for (tlwh,s) in zip(boxes_second,scores_second)]
        else:
            detections_second=[]
        second_tracked_stracks=[r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state==TrackState.Tracked or r_tracked_stracks[i].state==TrackState.Lost]
        dists=matching.iou_distance(second_tracked_stracks,detections_second)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, second_tracked_stracks,detections_second)
        logging.info(f"dists3: {dists}")
        matches,u_track_second,u_detection_second=matching.linear_assignment(dists,thresh=0.85)
        for itracked,idet in matches:
            track = second_tracked_stracks[itracked]
            det = detections_second[idet]
            
            if track.state==TrackState.Tracked:
                track.update(det,self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det,self.frame_id,new_id=False)
                refind_stracks.append(track)
        
        for it in u_track_second:
            track=second_tracked_stracks[it]
            if not track.state==TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                logging.info("Mark this track as lost")
                
        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections=[detections[i] for i in u_detection]
        
        dists=matching.iou_distance(unconfirmed,detections)
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, unconfirmed, detections)
        logging.info(f"dists4: {dists}")
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track=unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            logging.info(f"Remove this track: {removed_stracks}")
            
        """Step 4: Init new stracks"""
        for inew in u_detection:
            track=detections[inew]
            if track.score<self.det_thresh:
                continue
            track.activate(self.kalman_filter,self.frame_id)
            activated_stracks.append(track)
            logging.info("Activate new track_id")
        """Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id-track.end_frame>self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                logging.info("Remove this track because of exceed frame")
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        logging.info(f"Lost: {self.lost_stracks}")
        logging.info(f"Removed: {self.removed_stracks}")
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        logging.info(f"Before remove: {self.lost_stracks}")
        self.removed_stracks.extend(removed_stracks)
        logging.info(f"Before remove: {self.tracked_stracks}, {self.lost_stracks}")
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.tracked_stracks, _ = remove_duplicate_stracks(self.tracked_stracks, [])  # Only check within tracked
        # Don't remove duplicates between tracked and lost tracks
        logging.info(f"After: {self.tracked_stracks}")
        self.output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        bboxes=[]
        scores=[]
        ids=[]
        
        for track in self.output_stracks:
            if track.is_activated:
                logging.info(f"Track: {track}, state: {track.state}")
                
                track_bbox=track.tlbr
                bboxes.append([max(0,track_bbox[0]), max(0,track_bbox[1]), track_bbox[2], track_bbox[3]])
                scores.append(track.score)
                ids.append(track.track_id)
        return bboxes,scores,ids
    
    def get_switch_candidates(self):
        """
        Get the detected ID switch candidates
        
        Returns:
            Dictionary mapping track IDs to lists of frames with potential switches
        """
        return self.switch_candidates
    
    @staticmethod
    def reset_id():
        STrack.reset_id()
        
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[(t.track_id, t.start_frame)] = t  # Dùng tuple để phân biệt các lần track

    for t in tlistb:
        key = (t.track_id, t.start_frame)
        if key in stracks:
            del stracks[key]

    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.1)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain