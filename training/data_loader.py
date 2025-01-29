class MedicalTransform:
    def __call__(self, frames):
        # Ultrasound-specific augmentations
        frames += torch.randn_like(frames) * 0.05  # Speckle noise
        frames *= torch.FloatTensor(1).uniform_(0.8, 1.2)  # Gain variation
        
        # Temporal augmentations
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, [2])  # Reverse temporal
        
        # Spatial augmentations
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, [3])  # Horizontal flip
            
        return frames
