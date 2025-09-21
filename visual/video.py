# import argparse
# import json
# import cv2
# import os
# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm
# from transformers import ViTImageProcessor
# from PIL import Image

# def frame_to_input(frame, processor):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     inputs = processor(images=image, return_tensors="np")
#     return inputs["pixel_values"][0]

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", required=True, help="Path to .h5 model file")
#     parser.add_argument("--video_path", required=True, help="Input video")
#     parser.add_argument("--fps_sample", type=float, default=1.0, help="Frames per second to sample")
#     parser.add_argument("--threshold", type=float, default=0.5, help="Frame-level decision threshold")
#     parser.add_argument("--video_threshold", type=float, default=0.5, help="Video-level fake ratio threshold")
#     parser.add_argument("--output_report", required=True, help="JSON output file")
#     parser.add_argument("--output_annotated", help="Optional annotated video output")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
#     args = parser.parse_args()

#     # Load model
#     model = tf.keras.models.load_model(args.model_path)
#     processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

#     # Open video
#     cap = cv2.VideoCapture(args.video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     step = int(round(fps / args.fps_sample))
#     frames = []
#     frame_indices = []

#     idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if idx % step == 0:
#             frames.append(frame)
#             frame_indices.append(idx)
#         idx += 1
#     cap.release()

#     # Preprocess frames
#     inputs = np.array([frame_to_input(f, processor) for f in frames])

#     # Predict in batches
#     probs = []
#     for i in tqdm(range(0, len(inputs), args.batch_size), desc="Predicting"):
#         batch = inputs[i:i+args.batch_size]
#         preds = model.predict(batch)
#         probs.extend(preds[:,0])
#     probs = np.array(probs)

#     # Frame decisions
#     frame_labels = ["FAKE" if p >= args.threshold else "REAL" for p in probs]
#     fake_ratio = np.mean([p >= args.threshold for p in probs])
#     video_label = "FAKE" if fake_ratio >= args.video_threshold else "REAL"

#     # Save report
#     report = {
#         "video": args.video_path,
#         "video_label": video_label,
#         "video_fake_ratio": float(fake_ratio),
#         "frame_results": [
#             {"frame_index": int(fi), "prob_fake": float(p), "label": l}
#             for fi, p, l in zip(frame_indices, probs, frame_labels)
#         ]
#     }
#     with open(args.output_report, "w") as f:
#         json.dump(report, f, indent=2)

#     print(f"Analysis complete. Video label: {video_label}")
#     print(f"Report saved to {args.output_report}")

#     # Annotated video
#     if args.output_annotated:
#         cap = cv2.VideoCapture(args.video_path)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(args.output_annotated, fourcc, fps, (width, height))

#         frame_dict = dict(zip(frame_indices, zip(probs, frame_labels)))
#         idx = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if idx in frame_dict:
#                 prob, label = frame_dict[idx]
#                 text = f"{label} ({prob:.2f})"
#                 color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
#                 cv2.putText(frame, text, (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#             out.write(frame)
#             idx += 1
#         cap.release()
#         out.release()
#         print(f"Annotated video saved to {args.output_annotated}")


import argparse
import json
import cv2
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import ViTImageProcessor
from PIL import Image

def frame_to_input(frame, processor):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="np")
    return inputs["pixel_values"][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .h5 model file")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument("--fps_sample", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--threshold", type=float, default=0.5, help="Frame-level decision threshold")
    parser.add_argument("--video_threshold", type=float, default=0.5, help="Video-level fake ratio threshold")
    parser.add_argument("--output_report", required=True, help="JSON output file")
    parser.add_argument("--output_annotated", help="Optional annotated video output")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(fps / args.fps_sample))
    frames = []
    frame_indices = []

    print(f" Processing video: {args.video_path}")
    print(f" Original FPS: {fps}, Sampling every {step} frames")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
            frame_indices.append(idx)
        idx += 1
    cap.release()

    print(f"🎬 Total frames to analyze: {len(frames)}")

    # Preprocess frames
    inputs = np.array([frame_to_input(f, processor) for f in frames])

    # Predict in batches - FIXED PREDICTION LOGIC
    probs = []
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Predicting"):
        batch = inputs[i:i+args.batch_size]
        preds = model.predict(batch)
        
        # Convert logits to probabilities
        batch_probs = tf.nn.softmax(preds, axis=-1)
        fake_probs = batch_probs[:, 1].numpy()  # Index 1 = FAKE class
        probs.extend(fake_probs)
    
    probs = np.array(probs)

    # Frame decisions
    frame_labels = ["FAKE" if p >= args.threshold else "REAL" for p in probs]
    fake_ratio = np.mean([p >= args.threshold for p in probs])
    video_label = "FAKE" if fake_ratio >= args.video_threshold else "REAL"

    # Statistics
    fake_frames = sum(1 for label in frame_labels if label == "FAKE")
    real_frames = len(frame_labels) - fake_frames
    avg_fake_prob = np.mean(probs)

    # Enhanced report
    report = {
        "video": args.video_path,
        "video_label": video_label,
        "video_fake_ratio": float(fake_ratio),
        "statistics": {
            "total_frames_analyzed": len(frames),
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "average_fake_probability": float(avg_fake_prob),
            "max_fake_probability": float(np.max(probs)),
            "min_fake_probability": float(np.min(probs))
        },
        "parameters": {
            "fps_sample": args.fps_sample,
            "frame_threshold": args.threshold,
            "video_threshold": args.video_threshold,
            "batch_size": args.batch_size
        },
        "frame_results": [
            {
                "frame_index": int(fi), 
                "prob_fake": float(p), 
                "prob_real": float(1-p),
                "label": l,
                "confidence": float(p if l == "FAKE" else 1-p)
            }
            for fi, p, l in zip(frame_indices, probs, frame_labels)
        ]
    }
    
    # Save report
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)

    # Print results
    print(f"\n ANALYSIS RESULTS:")
    print(f"Video Label: {video_label}")
    print(f" Fake Ratio: {fake_ratio:.2%}")
    print(f" Frames analyzed: {len(frames)}")
    print(f" Fake frames: {fake_frames}")
    print(f" Real frames: {real_frames}")
    print(f" Average fake probability: {avg_fake_prob:.3f}")
    print(f" Report saved: {args.output_report}")

    # Annotated video - ENHANCED
    if args.output_annotated:
        print(f"\n🎬 Creating annotated video...")
        cap = cv2.VideoCapture(args.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output_annotated, fourcc, fps, (width, height))

        frame_dict = dict(zip(frame_indices, zip(probs, frame_labels)))
        idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if idx in frame_dict:
                prob, label = frame_dict[idx]
                # Enhanced annotation
                text1 = f"{label}"
                text2 = f"Confidence: {prob:.2%}" if label == "FAKE" else f"Confidence: {(1-prob):.2%}"
                
                color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
                
                # Main label
                cv2.putText(frame, text1, (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                # Confidence
                cv2.putText(frame, text2, (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Progress bar for fake probability
                bar_width = 200
                bar_height = 20
                bar_x, bar_y = 30, height - 60
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                # Probability bar
                fill_width = int(bar_width * prob)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
                # Bar text
                cv2.putText(frame, f"Fake Prob: {prob:.1%}", (bar_x, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(frame)
            idx += 1
            
        cap.release()
        out.release()
        print(f"Annotated video saved: {args.output_annotated}")