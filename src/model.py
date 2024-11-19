import cv2
import numpy as np
from ultralytics import YOLO

class Model:

    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def runModel(self, frame, target_class):
        frame_resized = cv2.resize(frame, (640, 480))

        results = self.model.predict(frame_resized, task='segment', conf=0.3)[0]

        masks = results.masks  # Mask data
        class_names = self.model.names  # Class names for the model
        boxes = results.boxes  # Bounding boxes, confidences, and class IDs
        
        final_mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
        detected_classes = []

        if masks is not None and boxes is not None:
            for mask, box in zip(masks.data, boxes):
                class_id = int(box.cls.cpu().numpy())  # Class ID
                class_name = class_names[class_id]  # Get the class name
                detected_classes.append(class_name)

                # Only process the "person" class (modify as needed)
                if class_name == target_class:
                    final_mask = np.maximum(final_mask, mask.cpu().numpy().astype(np.uint8))

        final_mask = cv2.resize(final_mask, (frame.shape[1], frame.shape[0]))
        print(final_mask.shape)

        # # Apply the mask on the original frame
        # segmented_frame = cv2.bitwise_and(frame, frame, mask=final_mask)

        return final_mask
    
    def applyMask(self, frame, mask, mask_type, mask_region):

        if mask_region == "Object":
            mask = 1 - mask

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        if mask_type == "Black":
            # do nothing since black is the default mask
            pass
        elif mask_type == "White":
            masked_frame[mask == 0] = 255
        elif mask_type == "Blur":
            masked_frame = cv2.GaussianBlur(frame, (15, 15), 3)
            masked_frame[mask == 1] = frame[mask == 1]

        return masked_frame