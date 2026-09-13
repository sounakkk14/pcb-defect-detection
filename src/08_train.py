import argparse
from pathlib import Path

from ultralytics import YOLO

project = Path(__file__).resolve().parents[1]

# Defaults reproduce runs/pcb_yolo_30epochs
parser = argparse.ArgumentParser(description="Train YOLO on the PCB defect dataset")
parser.add_argument("--model", default=str(project / "yolo26n.pt"))
parser.add_argument("--data", default=str(project / "pcb-defect-dataset" / "data.yaml"))
parser.add_argument("--name", default="pcb_yolo_30epochs")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--device", default="0")
parser.add_argument("--fraction", type=float, default=1.0, help="Use part of the train set (quick checks)")


# Windows needs the main guard for dataloader worker processes
if __name__ == "__main__":

    args = parser.parse_args()

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    print("Model :", args.model)
    print("Data  :", args.data)
    print("Run   :", project / "runs" / args.name)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(project / "runs"),
        name=args.name,
    )

    print("\nBest weights:", project / "runs" / args.name / "weights" / "best.pt")
