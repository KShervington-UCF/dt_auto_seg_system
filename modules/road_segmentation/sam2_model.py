import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

class SAM2Model:
    def __init__(self):
        self.model = self.load_model()
       
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

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Generate single prediction based on image
                masks, scores, logits = self.model.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                # Get the model's best mask
                mask_input = logits[np.argmax(scores), :, :]

                masks, scores, _ = self.model.predict(point_coords=input_points, point_labels=input_labels, mask_input=mask_input[None, :, :], multimask_output=False)

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]

            return masks[0], scores[0]
        except Exception as e:
            print(f'Error segmenting road\n{e}')
            raise(e)