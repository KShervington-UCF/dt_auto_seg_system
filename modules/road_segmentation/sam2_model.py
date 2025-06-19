import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
from pathlib import Path

class SAM2Model:
    def __init__(self):
        self.model = self.load_model()

        # Add to SAM2Model.__init__
        self.device = torch.device("cuda") if torch.cuda.is_available() else Exception("No GPU available")
        print(f"Using device: {self.device}")
       
    def load_model(self, model_name='facebook/sam2-hiera-large'):
        try:
            predictor = SAM2ImagePredictor.from_pretrained(model_name)

            return predictor
        except Exception as e:
            print(f'Model failed to load\n{e}')
            raise(e)
    
    def segment_road(self, image_path, input_points, input_labels):
        try:
            image = Image.open(image_path)
            image = np.array(image.convert('RGB'))

            self.model.set_image(image)

            # First generate multiple masks
            with torch.inference_mode():
                if torch.cuda.is_available():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        masks, scores, logits = self.model.predict(
                            point_coords=input_points, 
                            point_labels=input_labels, 
                            multimask_output=True
                        )
                else:
                    masks, scores, logits = self.model.predict(
                        point_coords=input_points, 
                        point_labels=input_labels, 
                        multimask_output=True
                    )

            # Sort masks by score
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            # Select highest scoring mask's logits
            best_mask_logits = logits[0, :, :]  # Take the highest scoring mask directly

            # Second prediction with the best mask input
            with torch.inference_mode():
                if torch.cuda.is_available():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        masks, scores, _ = self.model.predict(
                            point_coords=input_points, 
                            point_labels=input_labels, 
                            mask_input=best_mask_logits[None, :, :], 
                            multimask_output=False
                        )
                else:
                    masks, scores, _ = self.model.predict(
                        point_coords=input_points, 
                        point_labels=input_labels, 
                        mask_input=best_mask_logits[None, :, :], 
                        multimask_output=False
                    )

            # Sort final masks
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]

            return masks[0], scores[0]
        except Exception as e:
            print(f'Error segmenting road\n{e}')
            raise(e)