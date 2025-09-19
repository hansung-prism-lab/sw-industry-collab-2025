import os

class Get_path:
    base_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_path = os.path.join(base_path, "data")
    model_path = os.path.join(base_path, "models")
    encoder_path = os.path.join(model_path, "encoders")

