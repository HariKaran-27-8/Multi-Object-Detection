"""
YOLOv8 + DeepSORT real-time detection & tracking
Save as: main.py
Usage examples:
  python main.py --source 0                   # webcam
  python main.py --source input.mp4           # video file
  python main.py --source input.mp4 --output out.mp4 --show False
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# deep_sort_realtime (pip package)
from deep_sort_realtime.deepsort_tracker import DeepSort


# map COCO class indexes to names (short list—ultralytics has its own; this is enough for common classes)
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
    "cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","","dining table","","toilet","","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0 for webcam or path to video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model file (n/s/m/l/x)")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--show", type=lambda s: s.lower() in ['true','1','yes'], default=True, help="Whether to show live window")
    p.add_argument("--output", default=None, help="Path to save output video (optional)")
    return p.parse_args()


def xyxy_to_xywh(box):
    # input box = [x1,y1,x2,y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return [cx, cy, w, h]


def main():
    args = parse_args()

    # load model
    model = YOLO(args.model)  # Ultralyics YOLO object
    # create tracker
    tracker = DeepSort(max_age=30)

    # open source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.output:
        # prepare VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_input, (width, height))

    print("Starting. Press 'q' to quit.")

    prev = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optionally resize for speed (uncomment to reduce resolution)
        # frame = cv2.resize(frame, (960, 540))

        t0 = time.time()

        # Run detection (ultralytics returns iterable results; using stream=True reduces memory)
        results = model(frame, imgsz=640, conf=args.conf, iou=args.iou)  # single frame inference

        detections_for_tracker = []  # list of [bbox, confidence, class_name] where bbox = [x1,y1,x2,y2]

        for r in results:  # results usually contain one item for single image
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            # boxes is a Boxes object; iterate through each
            for box in boxes:
                # box.xyxy, box.conf, box.cls
                try:
                    xyxy = box.xyxy.cpu().numpy().flatten()  # [x1,y1,x2,y2]
                    conf = float(box.conf.cpu().numpy().item())
                    cls = int(box.cls.cpu().numpy().item())
                except Exception:
                    # fallback if shapes differ
                    xyxy = np.array(box.xyxy).flatten()
                    conf = float(box.conf)
                    cls = int(box.cls)

                # optionally filter classes (e.g., only persons and cars)
                # if COCO_NAMES[cls] not in ("person", "car"): continue

                # append to tracker inputs - format: [bbox, confidence, class_name]
                name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls)
                bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                detections_for_tracker.append([bbox, float(conf), name])

        # update tracker
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # draw detections/tracks
        for det in detections_for_tracker:
            bbox, score, name = det
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)
            cv2.putText(frame, f"{name} {score:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)

        # draw tracks with ID
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            l, t, r, b = map(int, ltrb)
            cls_name = track.det_class or ""  # sometimes available
            label = f"ID:{track_id} {cls_name}"
            cv2.rectangle(frame, (l, t), (r, b), (0, 200, 0), 2)
            cv2.putText(frame, label, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # fps overlay
        curr = time.time()
        fps = 1.0 / (curr - prev) if prev else 0.0
        prev = curr
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,255), 2)

        # show / save
        if args.show:
            cv2.imshow("YOLOv8 + DeepSORT", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
