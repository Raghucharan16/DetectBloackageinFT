import streamlit as st
import cv2
import torch
import numpy as np
from grad_cam import GradCAMX3D
from inference.preprocess import process_video

def main():
    st.title("Fallopian Tube Blockage Detection")
    
    # Load model
    model = GradCAMX3D()
    model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
    
    # File upload
    uploaded_file = st.file_uploader("Upload Ultrasound Video", type=["mp4", "avi"])
    
    if uploaded_file:
        # Save temp file
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_file.read())
            
        # Process video
        clips, original_frames = process_video("temp.mp4")
        
        # Inference
        preds = []
        heatmaps = []
        for clip in clips:
            with torch.no_grad():
                output = model(clip)
                pred = torch.softmax(output, dim=1)
                heatmap = model.generate_heatmap(clip, pred.argmax())
                preds.append(pred)
                heatmaps.append(heatmap)
                
        # Aggregate predictions
        final_pred = torch.mean(torch.stack(preds), dim=0)
        
        # Display results
        st.subheader("Prediction:")
        if final_pred[0][1] > 0.5:
            st.error(f"Blocked (Confidence: {final_pred[0][1]*100:.2f}%)")
        else:
            st.success(f"Normal (Confidence: {final_pred[0][0]*100:.2f}%)")
            
        # Show heatmap video
        heatmap_video = create_overlay_video(original_frames, heatmaps)
        st.video(heatmap_video)

def create_overlay_video(frames, heatmaps):
    # Create video with Grad-CAM overlay
    output_frames = []
    for frame, heat in zip(frames, heatmaps):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        heat = cv2.resize(heat, (224, 224))
        heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, heat, 0.3, 0)
        output_frames.append(blended)
    
    # Save as video
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (224,224))
    for f in output_frames:
        out.write(f)
    out.release()
    return open("output.mp4", "rb").read()

if __name__ == "__main__":
    main()
