import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

project = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Detect PCB defects in an image or a folder of images")
parser.add_argument("--weights", default=str(project / "runs" / "pcb_yolo_30epochs" / "weights" / "best.pt"))
parser.add_argument("--source", default=str(project / "pcb-defect-dataset" / "test" / "images"))
parser.add_argument("--output", default=str(project / "runs" / "pcb_predictions"))
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--device", default="0")


if __name__ == "__main__":

    args = parser.parse_args()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("PREDICTING")
    print("=" * 50)
    print("Weights:", args.weights)
    print("Source :", args.source)
    print("Output :", output_folder)

    model = YOLO(args.weights)

    detections = []
    image_count = 0
    defect_images = 0

    # stream=True processes one image at a time instead of holding all results in memory
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:

        image_name = Path(result.path).name
        image_count += 1

        # Save image with boxes drawn
        result.save(filename=str(output_folder / image_name))

        if len(result.boxes) > 0:
            defect_images += 1

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])

            detections.append({
                "image": image_name,
                "class": result.names[class_id],
                "confidence": round(float(box.conf[0]), 4),
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            })

    # Save all detections as CSV
    report_path = output_folder / "detections.csv"

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "class", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(detections)

    print("\nImages processed     :", image_count)
    print("Images with defects  :", defect_images)
    print("Total defects found  :", len(detections))
    print("Detections CSV       :", report_path)
