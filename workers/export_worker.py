# -*- coding: utf-8 -*-
"""
TensorRT 匯出子行程：把 .pt 轉成 Jetson 上跑的 .engine（預設 FP16）。

獨立成一個行程有兩個好處：
  1. 訓練行程先結束、記憶體全部歸還，才開始建 engine（TensorRT 建圖本身很吃記憶體）。
  2. 匯出失敗不會影響已經訓練好的 .pt。

用法：python3 export_worker.py <job.json>
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
    src = job["model"]
    imgsz = int(job.get("imgsz", 640))
    half = bool(job.get("half", True))
    emit("start", model=os.path.basename(src), imgsz=imgsz, half=half)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        emit("error", message="沒有可用的 CUDA 裝置，TensorRT 匯出需要 GPU")
        sys.exit(1)
    try:
        import tensorrt
        emit("info", message=f"TensorRT {tensorrt.__version__}")
    except ImportError:
        emit("error", message="找不到 tensorrt 套件，請確認 JetPack 的 TensorRT 已安裝")
        sys.exit(1)

    emit("info", message="開始建立 engine（TensorRT 建圖會跑一陣子，請耐心等）")
    model = YOLO(src)
    # workspace 是 TensorRT 建圖時的記憶體上限（GB）。這版 ultralytics 會直接拿它去
    # 乘 1<<30，傳 None 會爆 TypeError，所以一定要給數字；Jetson 記憶體小，預設 2GB。
    kw = {"format": "engine", "half": half, "imgsz": imgsz, "device": 0, "verbose": False,
          "workspace": float(job.get("workspace") or 2)}
    out = model.export(**kw)
    out = str(out)
    if not os.path.exists(out):
        emit("error", message=f"匯出結束但找不到檔案：{out}")
        sys.exit(1)

    dest = os.path.join(job["models_dir"], job["name"] + ".engine")
    os.makedirs(job["models_dir"], exist_ok=True)
    if os.path.abspath(out) != os.path.abspath(dest):
        shutil.move(out, dest)

    # Ultralytics 是 PT → ONNX → engine，中間的 .onnx 沒有用處又佔 10MB 左右，清掉
    onnx_tmp = os.path.splitext(src)[0] + ".onnx"
    if os.path.exists(onnx_tmp):
        try:
            os.remove(onnx_tmp)
            emit("info", message="已清除中間產物 " + os.path.basename(onnx_tmp))
        except OSError:
            pass

    # 記下建立條件：engine 綁定 GPU / TensorRT 版本 / 輸入尺寸，之後推論要對得上
    meta = {"source": os.path.basename(src), "imgsz": imgsz, "half": half,
            "tensorrt": tensorrt.__version__, "device": torch.cuda.get_device_name(0),
            "size_mb": round(os.path.getsize(dest) / 1048576, 1)}
    with open(dest + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    emit("done", path=dest, **meta)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit("error", message=str(e), trace=traceback.format_exc()[-1500:])
        sys.exit(1)
