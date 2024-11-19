import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2


class CameraMasker():

    def __init__(self, model):
        self.model = model
        pass

    def generateImageMasker(self, selected_class, selected_mask, selected_mask_region):
        
        st.header("Camera Masker")

        uploaded_image = st.camera_input("Take a picture using your camera:")

        if uploaded_image is not None:
            # Read the image as a PIL object
            image = Image.open(uploaded_image)
            
            # running masking
            image = np.array(image)

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            mask = self.model.runModel(image, selected_class)
            masked_img = self.model.applyMask(image, mask, selected_mask, selected_mask_region)

            # converting back to PIL image
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(masked_img)

            # displaying image with Streamlit
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Save the uploaded image to a buffer (in memory)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Provide a download button
            st.download_button(
                label="Download Image",
                data=img_buffer,
                file_name="masked_image.png",
                mime="image/png"
            )


