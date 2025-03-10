import os
import tensorflow as tf
from tensorflow import keras

class ModelLoader:
    def __init__(self):
        # Get the directory of the script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = None
        
    def check_gpu(self):
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("GPU is Available!")
            return True
        return False
        
    # Model currently not shared but can be accessed personally at https://ucf-my.sharepoint.com/:u:/r/personal/ky455244_ucf_edu/Documents/auto-seg-system-files/road_surface_classifier_152V2_92.h5?csf=1&web=1&e=LmufH3
    def load_model(self, model_name='road_surface_classifier_152V2_92_weighted.h5'):
        # Get the absolute path of the model
        model_path = os.path.join(self.script_dir, 'model', model_name)
        
        try:
            if not self.check_gpu():
                raise Exception("No GPU available")
                
            self.model = keras.models.load_model(model_path)
            print('Model loaded successfully!')
            return self.model
            
        except Exception as e:
            print(f'Model failed to load\n{e}')
            raise