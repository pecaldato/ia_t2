import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import cv2


class VideoMasker():

    def __init__(self, model):
        self.model = model
        pass

    def generateImageMasker(self, selected_class, selected_mask, selected_mask_region):
        
        st.header("Image Masker")
        st.info("Select an video and choose a mask type to apply.")

        # File uploader for MP4 videos
        uploaded_file = st.file_uploader("", type=["mp4", ".mov", ".avi"])

        if uploaded_file is not None:

            # Save the uploaded file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

            
             # Read and process the video using OpenCV
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Error: Unable to open the video file.")
            else:
                st.write("Processing the uploaded video...")

                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

                # Prepare to save the processed video
                processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
                out = cv2.VideoWriter(processed_video_path, fourcc, frame_rate, (frame_width, frame_height))

                # Display the video frames
                stframe = st.empty()  # Streamlit container for video frames

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    mask = self.model.runModel(frame, selected_class)
                    masked_img = self.model.applyMask(frame, mask, selected_mask, selected_mask_region)

                    # Save the processed frame
                    out.write(masked_img)

                    # Display the current frame
                    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
                    frame_image = Image.fromarray(masked_img)
                    stframe.image(frame_image, caption="Processing Video...", use_container_width=True)

                # Release resources
                cap.release()
                out.release()

                # Provide a download button for the processed video
                with open(processed_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

                # Clean up the temporary video files
                os.remove(video_path)
                os.remove(processed_video_path)


