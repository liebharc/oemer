import json
import os

def save_model(model, metadata, model_path):
    """Save model and metadata"""
    model.export(model_path)  # Creates a folder with the model, we now add metadata
    _write_text_to_file(model.to_json(), os.path.join(model_path, "arch.json"))  # Save model architecture for documentation
    _write_text_to_file(json.dumps(metadata), os.path.join(model_path, "meta.json"))

def load_model(model_path):
    """Load model and metadata"""
    import tensorflow as tf
    model = tf.saved_model.load(model_path)
    with open(os.path.join(model_path, "meta.json"), "r") as f:
        metadata = json.loads(f.read())
    return model, metadata

def _write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)