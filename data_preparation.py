import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

class TubeBlockageDataPreparator:
    """
    Handles data preparation for fallopian tube blockage detection from a two-folder structure.
    This class processes videos from 'normal' and 'blocked' folders and prepares them for model training.
    """
    def __init__(self, base_path, num_frames=64, frame_height=224, frame_width=224):
        """
        Initialize the data preparator with paths and video specifications.
        
        Args:
            base_path: Root directory containing 'normal' and 'blocked' folders
            num_frames: Number of frames to extract from each video
            frame_height: Height to resize frames to
            frame_width: Width to resize frames to
        """
        self.base_path = Path(base_path)
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Define paths for normal and blocked videos
        self.normal_path = self.base_path / 'normal'
        self.blocked_path = self.base_path / 'blocked'
        
        # Validate directory structure
        self._validate_directories()

    def _validate_directories(self):
        """
        Ensures the required directory structure exists and contains videos.
        Provides helpful error messages if the structure isn't correct.
        """
        if not self.normal_path.exists():
            raise FileNotFoundError(f"Normal videos folder not found at {self.normal_path}")
        if not self.blocked_path.exists():
            raise FileNotFoundError(f"Blocked videos folder not found at {self.blocked_path}")
            
        # Check for supported video files
        self.video_extensions = ('.mp4', '.avi', '.mov')
        normal_videos = self._get_video_files(self.normal_path)
        blocked_videos = self._get_video_files(self.blocked_path)
        
        print(f"Found {len(normal_videos)} normal videos and {len(blocked_videos)} blocked videos")

    def _get_video_files(self, directory):
        """
        Gets all video files from a directory.
        
        Args:
            directory: Path to search for videos
            
        Returns:
            List of video file paths
        """
        return [
            f for f in directory.glob('**/*') 
            if f.suffix.lower() in self.video_extensions
        ]

    def process_video(self, video_path):
        """
        Processes a single video file by extracting and preprocessing frames.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Preprocessed frames as a numpy array
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Calculate frame sampling rate to get desired number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_rate = max(total_frames // self.num_frames, 1)
        
        frame_count = 0
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sampling_rate == 0:
                # Preprocess frame
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        
        # Handle videos shorter than target length
        while len(frames) < self.num_frames:
            frames.append(np.zeros_like(frames[0]))
            
        return np.array(frames)

    def prepare_datasets(self, test_size=0.2, validation_size=0.2):
        """
        Prepares train, validation, and test datasets from the video folders.
        
        Args:
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Train, validation, and test datasets as TensorFlow datasets
        """
        # Get all video files
        normal_videos = self._get_video_files(self.normal_path)
        blocked_videos = self._get_video_files(self.blocked_path)
        
        # Create labels (0 for blocked, 1 for normal)
        all_videos = blocked_videos + normal_videos
        all_labels = [0] * len(blocked_videos) + [1] * len(normal_videos)
        
        # Split into train+val and test sets
        train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
            all_videos, all_labels, test_size=test_size, random_state=42, stratify=all_labels
        )
        
        # Further split train+val into train and validation sets
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            train_val_videos, train_val_labels, 
            test_size=validation_size, 
            random_state=42,
            stratify=train_val_labels
        )
        
        print(f"Training set: {len(train_videos)} videos")
        print(f"Validation set: {len(val_videos)} videos")
        print(f"Test set: {len(test_videos)} videos")
        
        return (
            self.create_dataset(train_videos, train_labels, is_training=True),
            self.create_dataset(val_videos, val_labels, is_training=False),
            self.create_dataset(test_videos, test_labels, is_training=False)
        )

    def create_dataset(self, video_paths, labels, batch_size=8, is_training=True):
        """
        Creates a TensorFlow dataset from video paths and labels.
        
        Args:
            video_paths: List of paths to video files
            labels: List of corresponding labels
            batch_size: Size of batches to create
            is_training: Whether this is a training dataset
            
        Returns:
            TensorFlow dataset
        """
        def load_and_preprocess(video_path, label):
            # Convert video path from tensor to string
            video_path = video_path.numpy().decode('utf-8')
            # Process video
            frames = self.process_video(video_path)
            # Create slow and fast pathway inputs for SlowFast network
            slow_frames = frames[::4]  # Every 4th frame
            fast_frames = frames
            return (slow_frames, fast_frames), label

        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
        
        # Map preprocessing function
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                load_and_preprocess,
                [x, y],
                [(tf.float32, tf.float32), tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            dataset = dataset.shuffle(1000)
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
