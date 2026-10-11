# -*- coding: utf-8 -*-
"""
測試集評估子行程：對「訓練／驗證／測試切分」產生的 held-out test 集跑一次
model.val(split="test")——不是訓練，只驗證，通常幾十秒到幾分鐘內結束。

用法：python3 eval_worker.py <job.json>
stdout：只印一行 "@@{json}" 事件（done / error），server.py 用 subprocess.run() 同步呼叫、
直接解析 stdout，不需要像訓練那樣輪詢進度。
"""
import json
import sys
import traceback


def emit(event, **kw):
    kw["event"] = event
    print("@@" + json.dumps(kw, ensure_ascii=False), flush=True)


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))

    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(job["model"])
    results = model.val(data=job["data"], split="test", imgsz=int(job.get("imgsz", 640)),
                        device=device, plots=False, verbose=False)

    rd = getattr(results, "results_dict", {}) or {}
    p = float(rd.get("metrics/precision(B)", 0) or 0)
    r = float(rd.get("metrics/recall(B)", 0) or 0)
    metrics = {
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0) or 0), 4),
        "mAP50-95": round(float(rd.get("metrics/mAP50-95(B)", 0) or 0), 4),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
    }
    emit("done", metrics=metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit("error", message=str(e), trace=traceback.format_exc()[-1500:])
        sys.exit(1)
