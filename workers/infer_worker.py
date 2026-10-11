# -*- coding: utf-8 -*-
"""
推論子行程：常駐但會自己過期。閒置超過 IDLE_TIMEOUT 秒就自行結束，
把模型與 torch 佔的記憶體還給系統；下次要用時 server 會重新叫起來。

協定：stdin 一行一個 JSON request，stdout 一行一個 JSON response。
  {"cmd":"predict","model":"/path/x.pt","path":"/img.jpg","conf":0.25}
  {"cmd":"bye"}
"""
import json
import os
import sys
import threading
import time

# 協定通道：把原本的 fd 1 複製一份專門用來回 JSON，再把 fd 1 導到 stderr。
# 這樣不管是 ultralytics 的 LOGGER 還是 TensorRT 的 C 層訊息（例如載入 engine 時的
# "Loading xxx.engine for TensorRT inference..."）都不會混進協定，把回應弄壞。
_PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr


def reply(obj):
    _PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTO.flush()


IDLE_TIMEOUT = float(os.environ.get("LITE_INFER_IDLE", "180"))
_last = time.time()
_model = None
_model_path = ""


def _watchdog():
    while True:
        time.sleep(5)
        if time.time() - _last > IDLE_TIMEOUT:
            os._exit(0)          # 閒置過久 → 直接結束，釋放全部記憶體


def _load(path):
    global _model, _model_path
    if _model is None or _model_path != path:
        from ultralytics import YOLO
        if path.endswith(".engine"):
            # .engine 的 task 一般寫在 TensorRT 檔案的 metadata 裡；讀不到就當偵測模型
            try:
                _model = YOLO(path)
            except Exception:
                _model = YOLO(path, task="detect")
        else:
            _model = YOLO(path)
        _model_path = path
    return _model


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    reply({"ok": True, "ready": True})
    global _last
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        _last = time.time()
        try:
            req = json.loads(line)
            if req.get("cmd") == "bye":
                reply({"ok": True, "bye": True})
                return
            model = _load(req["model"])
            kw = {"source": req["path"], "conf": float(req.get("conf", 0.25)),
                  "verbose": False, "save": False}
            if req["model"].endswith(".engine"):
                # engine 的輸入尺寸在建圖時就固定了，交給 Ultralytics 讀 metadata，
                # 這裡不要再指定 imgsz / half，避免和 engine 對不上
                pass
            else:
                kw["imgsz"] = int(req.get("imgsz", 640))
                if req.get("half"):
                    kw["half"] = True
            res = model.predict(**kw)[0]
            names = res.names
            dets = []
            if getattr(res, "obb", None) is not None and res.obb is not None and len(res.obb):
                for b in res.obb:
                    xy = b.xyxyxyxy[0].tolist()
                    dets.append({"type": "obb", "cls": names[int(b.cls)], "conf": round(float(b.conf), 4),
                                 "points": [{"x": p[0], "y": p[1]} for p in xy]})
            elif getattr(res, "masks", None) is not None and res.masks is not None and len(res.masks):
                for poly, box in zip(res.masks.xy, res.boxes):
                    dets.append({"type": "seg", "cls": names[int(box.cls)], "conf": round(float(box.conf), 4),
                                 "points": [{"x": float(p[0]), "y": float(p[1])} for p in poly[::3]]})
            elif getattr(res, "probs", None) is not None and res.probs is not None:
                top = int(res.probs.top1)
                dets.append({"type": "cls", "cls": names[top], "conf": round(float(res.probs.top1conf), 4)})
            elif res.boxes is not None:
                for b in res.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    dets.append({"type": "bbox", "cls": names[int(b.cls)], "conf": round(float(b.conf), 4),
                                 "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
            h, w = res.orig_shape
            reply({"ok": True, "dets": dets, "width": w, "height": h,
                   "speed": {k: round(v, 1) for k, v in (res.speed or {}).items()}})
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
