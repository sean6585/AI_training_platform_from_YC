# -*- coding: utf-8 -*-
"""
訓練子行程：只有這裡會 import torch / ultralytics。
訓練結束即行程結束，記憶體全數歸還作業系統（主 server 永遠不碰 torch）。

用法：python3 train_worker.py <job.json>
stdout：Ultralytics 原始 log 直接輸出；本程式的事件以 "@@{json}" 單行輸出供 server 解析。
"""
import json
import os
import shutil
import sys
import traceback


def emit(event, **kw):
    kw["event"] = event
    print("@@" + json.dumps(kw, ensure_ascii=False), flush=True)


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    emit("start", model=job["model"], data=job["data"], epochs=job["epochs"])

    import torch                                    # noqa: F401  （先載入才拿得到 cuda 狀態）
    from ultralytics import YOLO

    device = job.get("device") or ("0" if torch.cuda.is_available() else "cpu")
    emit("device", device=str(device), cuda=torch.cuda.is_available())

    model = YOLO(job["model"])
    task = model.task
    emit("task", task=task)

    def on_epoch_end(trainer):
        m = {k.split("/")[-1].replace("(B)", ""): round(float(v), 4)
             for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        emit("epoch", epoch=int(trainer.epoch) + 1, epochs=int(trainer.epochs), metrics=m)

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    aug = job.get("aug") or {}
    results = model.train(
        data=job["data"],
        epochs=int(job["epochs"]),
        imgsz=int(job.get("imgsz", 640)),
        batch=int(job.get("batch", 8)),
        device=device,
        workers=int(job.get("workers", 2)),
        project=job["project"],
        name=job["name"],
        exist_ok=True,
        pretrained=True,
        plots=True,
        val=True,
        cache=False,                                  # Jetson 記憶體有限，不把資料集塞進 RAM
        **aug,
    )

    save_dir = str(results.save_dir if hasattr(results, "save_dir") else
                   os.path.join(job["project"], job["name"]))

    # 取最終指標（detect/seg/obb 用 box mAP；classify 用 top1）
    metrics = {}
    try:
        rd = results.results_dict if hasattr(results, "results_dict") else {}
        p = float(rd.get("metrics/precision(B)", 0) or 0)
        r = float(rd.get("metrics/recall(B)", 0) or 0)
        metrics = {
            "mAP": round(float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50(B)", 0)) or 0), 4),
            "mAP50": round(float(rd.get("metrics/mAP50(B)", 0) or 0), 4),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
            "Accuracy_Top1": round(float(rd.get("metrics/accuracy_top1", 0) or 0), 4),
        }
    except Exception:
        emit("warn", message="無法解析最終指標")

    metrics["dataset"] = job.get("dataset_stats", {})
    metrics["base_model"] = os.path.basename(job["model"])
    metrics["task"] = task
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False)

    # best.pt → models/<name>.pt（與主程式相同慣例，兩邊的「模型評估 / 推論」都吃得到）
    best = os.path.join(save_dir, "weights", "best.pt")
    out_model = ""
    if os.path.exists(best):
        suffix = {"classify": "-cls", "segment": "-seg", "obb": "-obb"}.get(task, "")
        out_model = os.path.join(job["models_dir"], f"{job['name']}{suffix}.pt")
        os.makedirs(job["models_dir"], exist_ok=True)
        shutil.copy2(best, out_model)

    emit("done", save_dir=save_dir, model_path=out_model, metrics=metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit("error", message=str(e), trace=traceback.format_exc()[-2000:])
        sys.exit(1)
