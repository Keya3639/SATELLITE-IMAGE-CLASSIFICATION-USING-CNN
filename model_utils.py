import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

# ✅ Load saved model + class names
def load_saved(model_path, class_names_path):
    model = load_model(model_path)
    class_names = np.load(class_names_path)
    return model, class_names

# ✅ Predict top-2 classes
def predict_top2(model, class_names, pil_image, img_size=(64, 64)):
    # convert PIL → array
    img = pil_image.resize(img_size)
    img = img.convert("RGB")  # ensure RGB

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # prediction
    preds = model.predict(arr)[0]

    # get top-2 indices
    top2_idx = preds.argsort()[-2:][::-1]

    # return class names + probabilities
    top2 = [(class_names[i], float(preds[i])) for i in top2_idx]
    return top2
 
