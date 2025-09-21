# import argparse
# import json
# import tensorflow as tf
# from transformers import ViTImageProcessor
# from PIL import Image
# import numpy as np

# def load_and_preprocess(image_path, processor):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="np")
#     return inputs["pixel_values"][0]

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", required=True, help="Path to .h5 model file")
#     parser.add_argument("--image_path", required=True, help="Path to input image")
#     parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default=0.5)")
#     args = parser.parse_args()

#     # Load model
#     model = tf.keras.models.load_model(args.model_path)

#     # Preprocess
#     processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
#     pixel_values = load_and_preprocess(args.image_path, processor)
#     pixel_values = np.expand_dims(pixel_values, axis=0)

#     # Predict
#     prob_fake = float(model.predict(pixel_values)[0][0])
#     is_fake = prob_fake >= args.threshold

#     result = {
#         "image": args.image_path,
#         "prob_fake": prob_fake,
#         "predicted_label": "FAKE" if is_fake else "REAL",
#         "threshold": args.threshold
#     }

#     print(json.dumps(result, indent=2))

import argparse
import json
import tensorflow as tf
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np

def load_and_preprocess(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="np")
    return inputs["pixel_values"][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .h5 model file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default=0.5)")
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Preprocess
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    pixel_values = load_and_preprocess(args.image_path, processor)
    pixel_values = np.expand_dims(pixel_values, axis=0)

    # Predict - FIXED LINE
    predictions = model.predict(pixel_values)
    logits = predictions[0]
    probs = tf.nn.softmax(logits)
    prob_fake = float(probs[1])  # Index 1 = FAKE class
    prob_real = float(probs[0])  # Index 0 = REAL class
    
    is_fake = prob_fake >= args.threshold

    result = {
        "image": args.image_path,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "predicted_label": "FAKE" if is_fake else "REAL",
        "confidence": prob_fake if is_fake else prob_real,
        "threshold": args.threshold
    }

    print(json.dumps(result, indent=2))