# reid2024
A complete end-to-end reid system with edge-server architecture

All the files needed for running is under reid-2024/app: 
1. deeputils: code for ByteTrack implementation, including:
   a. ByteTrack implementation in base_track and bytetracker.py
   b. Kalman_filter for predict next position
   c. matching.py for match function, including match by features and match by IOU
   d. linear_assignment to assign the cost

2. edge_device: setup for sending data from edge device, including detection info and other metadatas.
3. models: some of the models for reid. You should git clone the torchreid repo for better use.
4. server: including the main implementation, including:
    a. Receive input
    b. Extracting features
    c. Tracking
    d. Update ID
5. offline_processing: code for processing offline, including:
    a. read data from tracking results
    b. detect id switches
    c. Match by cluster different tracklets
    d. Matching across cameras


   
