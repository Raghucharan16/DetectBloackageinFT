import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_hub as hub

# 1. Data Processing Utilities
class VideoDataProcessor:
    """
    Handles video data preprocessing for both I3D and SlowFast networks.
    Includes frame extraction, resizing, and normalization.
    """
    def __init__(self, frame_height=224, frame_width=224, num_frames=64):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames

    def load_video(self, video_path):
        """
        Loads and preprocesses a single video file.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame sampling rate
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
                frame = frame / 255.0  # Normalize
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        
        # Handle videos shorter than target length
        while len(frames) < self.num_frames:
            frames.append(np.zeros_like(frames[0]))
            
        return np.array(frames)

    def prepare_for_slowfast(self, frames):
        """
        Prepares frames for SlowFast network by creating slow and fast pathways
        """
        # Slow pathway: every 4th frame
        slow_frames = frames[::4]
        # Fast pathway: all frames
        fast_frames = frames
        
        return slow_frames, fast_frames

# 2. Dataset Creation
class VideoDatasetBuilder:
    """
    Creates TensorFlow datasets for training and validation
    """
    def __init__(self, processor):
        self.processor = processor

    def create_dataset(self, video_paths, labels, batch_size=8, is_training=True):
        """
        Creates a tf.data.Dataset from video paths and labels
        """
        def load_and_preprocess(video_path, label):
            frames = self.processor.load_video(video_path.decode())
            slow_frames, fast_frames = self.processor.prepare_for_slowfast(frames)
            return (slow_frames, fast_frames), label

        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
        
        # Apply preprocessing
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                load_and_preprocess,
                [x, y],
                [tf.float32, tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if is_training:
            dataset = dataset.shuffle(1000)

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 3. I3D Model Implementation
class I3DModel(tf.keras.Model):
    """
    Implementation of I3D model with pre-trained weights
    """
    def __init__(self, num_classes=2):
        super(I3DModel, self).__init__()
        
        # Load pre-trained I3D base
        self.base_model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
        
        # Add custom layers for fine-tuning
        self.global_pool = layers.GlobalAveragePooling3D()
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Process through I3D base
        x = self.base_model(inputs)
        x = self.global_pool(x)
        x = self.dropout(x)
        return self.classifier(x)

# 4. SlowFast Model Implementation
class SlowFastModel(tf.keras.Model):
    """
    Implementation of SlowFast network with pre-trained ResNet backbone
    """
    def __init__(self, num_classes=2):
        super(SlowFastModel, self).__init__()
        
        # Create slow and fast pathways
        self.slow_pathway = self._create_pathway(stride=4)
        self.fast_pathway = self._create_pathway(stride=1)
        
        # Fusion and classification layers
        self.fusion = layers.Concatenate()
        self.classifier = tf.keras.Sequential([
            layers.GlobalAveragePooling3D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    def _create_pathway(self, stride):
        """
        Creates a pathway using ResNet50 as backbone
        """
        backbone = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False
        )
        
        # Freeze early layers
        for layer in backbone.layers[:100]:
            layer.trainable = False
            
        return tf.keras.Sequential([
            layers.TimeDistributed(backbone),
            layers.Conv3D(256, kernel_size=(3, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs):
        slow_input, fast_input = inputs
        
        # Process through pathways
        slow_features = self.slow_pathway(slow_input)
        fast_features = self.fast_pathway(fast_input)
        
        # Combine features
        combined = self.fusion([slow_features, fast_features])
        
        # Classification
        return self.classifier(combined)

# 5. Training Configuration
class ModelTrainer:
    """
    Handles model training and evaluation
    """
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.learning_rate = learning_rate

    def configure_training(self):
        """
        Configures training parameters
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

    def train(self, train_dataset, val_dataset, epochs=50):
        """
        Trains the model with appropriate callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_auc',
                save_best_only=True
            )
        ]

        return self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

# 6. Main Training Pipeline
def main():
    # Initialize video processor
    processor = VideoDataProcessor(frame_height=224, frame_width=224, num_frames=64)
    # Initialize data preparator
    data_prep = TubeBlockageDataPreparator(
        base_path='path/to/your/data/folder'  
    )
    # Get prepared datasets
    train_dataset, val_dataset, test_dataset = data_prep.prepare_datasets()
  
    
    # Train I3D Model
    i3d_model = I3DModel(num_classes=2)
    i3d_trainer = ModelTrainer(i3d_model, learning_rate=1e-4)
    i3d_trainer.configure_training()
    i3d_history = i3d_trainer.train(train_dataset, val_dataset)
    
    # Train SlowFast Model
    slowfast_model = SlowFastModel(num_classes=2)
    slowfast_trainer = ModelTrainer(slowfast_model, learning_rate=1e-4)
    slowfast_trainer.configure_training()
    slowfast_history = slowfast_trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
