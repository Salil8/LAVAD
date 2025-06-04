# Step 2: Create temporal windows for each frame of a video centered around the frame

class Temporal_Window:
    def __init__(self, N): # Only give temporal window size
        # self.video_captions = video_captions # list of captions for each frame of the video (in order)
        self.N = N # window size
        # self.frame_paths = frame_paths # list of paths to frames of a video (in order)

    def generate_temporal_captions(self, video_caption):
        '''To create an (len(video_caption) x N) list where i-th entry of the list 
        is a list of N captions centered at index i
        
        Necessary padding is also done for initial and final N-1 frames'''
        # padding at beginning
        for i in range(int(self.N/2)):
            video_caption = ['*'] + video_caption

        # padding at end
        for i in range(int((self.N-1)/2)):
            video_caption = video_caption + ['*']

        final_list = []
        for i in range(len(video_caption)-self.N+1): # i is caption index
            temp_list = [] #for each frame
            for j in range(self.N):
                temp_list.append(video_caption[i+j])
            final_list.append(temp_list)
                
        return final_list
    
    def generate_temporal_video(self, frame_paths):
        '''To create an (len(frame_paths) x N) list where i-th entry of the list 
        is a list of N frames (paths) centered at index i
        
        Necessary padding is also done for initial and final N-1 frames'''
        # padding at beginning
        for i in range(int(self.N/2)):
            frame_paths = ['*'] + frame_paths

        # padding at end
        for i in range(int((self.N-1)/2)):
            frame_paths = frame_paths + ['*']

        temporal_video = []
        for i in range(len(frame_paths)-self.N+1): # i is frame index
            temp_list = [] #for each frame
            for j in range(self.N):
                temp_list.append(frame_paths[i+j])
            temporal_video.append(temp_list)

        return temporal_video