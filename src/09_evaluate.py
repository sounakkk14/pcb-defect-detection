import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

project = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split")
parser.add_argument("--weights", default=str(project / "runs" / "pcb_yolo_30epochs" / "weights" / "best.pt"))
parser.add_argument("--data", default=str(project / "pcb-defect-dataset" / "data.yaml"))
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--name", default="pcb_eval_test")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--device", default="0")


if __name__ == "__main__":

    args = parser.parse_args()

    print("=" * 50)
    print(f"EVALUATING ON {args.split.upper()}")
    print("=" * 50)
    print("Weights:", args.weights)

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project / "runs"),
        name=args.name,
        workers=2,
    )

    precision, recall, map50, map50_95 = metrics.mean_results()

    print("\n" + "=" * 50)
    print("OVERALL")
    print("=" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map50_95:.4f}")

    # Per-class table
    rows = metrics.summary()

    print("\n" + "=" * 50)
    print("PER CLASS")
    print("=" * 50)
    print(f"{'Class':<17}{'Inst':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'50-95':>8}")

    for row in rows:
        print(
            f"{row['Class']:<17}{int(row['Instances']):>6}"
            f"{row['Box-P']:>8.3f}{row['Box-R']:>8.3f}"
            f"{row['mAP50']:>8.3f}{row['mAP50-95']:>8.3f}"
        )

    # Save per-class results as CSV
    report_folder = project / "src" / "reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / f"{args.name}_per_class.csv"

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPlots saved to :", metrics.save_dir)
    print("CSV saved to   :", report_path)
