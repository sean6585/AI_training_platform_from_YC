# -*- coding: utf-8 -*-
"""
DefectMaster Lite — 輕量版 AI 訓練平台（Jetson 專用）

設計原則（相對於 app_for_offline.py）：
  1. 主行程「永不」import torch / ultralytics。訓練與推論都丟到子行程，
     結束或閒置即回收記憶體。主 server 常駐約 40MB。
  2. 前端不用 React / Babel / Tailwind CDN，改純 JS + CSS，省掉瀏覽器端每次
     載入都要即時編譯 364KB JSX 的成本。
  3. 資料集只在 server 端保存「檔案路徑清單」，前端分頁向後端要；圖片清單、
     標註都不會整包塞進瀏覽器記憶體。
  4. 訓練不複製資料集：外部 data.yaml 直接餵給 Ultralytics；資料夾模式則產生
     train.txt / val.txt（絕對路徑清單）＋ data.yaml，零複製完成 8:2 拆分。

輸出與 weights 目錄預設沿用同一包的 DefectMaster_Output / weights，
所以在這裡訓練出來的模型，主程式的「模型評估 / 推論」也看得到。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import licensing

# =========================
# 路徑
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(APP_DIR)                       # 與主程式同一包
WEB_DIR = os.path.join(APP_DIR, "web")
WORKER_DIR = os.path.join(APP_DIR, "workers")

WEIGHTS_DIR = next((d for d in (os.path.join(PKG_DIR, "weights"), os.path.join(APP_DIR, "weights"))
                    if os.path.isdir(d)), os.path.join(PKG_DIR, "weights"))
OUTPUT_DIR = os.path.join(PKG_DIR, "DefectMaster_Output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATASETS_DIR = os.path.join(OUTPUT_DIR, "datasets")
RUNS_DIR = OUTPUT_DIR                                     # 訓練輸出與主程式同層，評估頁共用
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
THUMB_DIR = os.path.join(OUTPUT_DIR, ".thumb_cache")
JOB_DIR = os.path.join(OUTPUT_DIR, ".lite_jobs")
BACKUP_DIR = os.path.join(OUTPUT_DIR, ".label_backups")
for d in (OUTPUT_DIR, MODELS_DIR, DATASETS_DIR, LOG_DIR, THUMB_DIR, JOB_DIR, BACKUP_DIR):
    os.makedirs(d, exist_ok=True)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
LOG_FILE = os.path.join(LOG_DIR, "lite_server.log")


def log(msg, level="INFO"):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def load_config():
    cfg_path = os.path.join(APP_DIR, "config.json")
    cfg = {"port": 8002, "host": "0.0.0.0", "infer_idle_seconds": 180}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg.update(json.load(f) or {})
        except Exception as e:
            log(f"config.json 讀取失敗，使用預設值：{e}", "WARNING")
    return cfg


CONFIG = load_config()
LICENSED, LICENSE_MSG, MACHINE_MAC = licensing.check_license([PKG_DIR, APP_DIR])

app = FastAPI(title="DefectMaster Lite")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.middleware("http")
async def license_gate(request: Request, call_next):
    open_paths = ("/", "/health", "/license/status", "/sysinfo")
    if request.url.path in open_paths or request.url.path.startswith("/web"):
        return await call_next(request)
    if not LICENSED:
        return JSONResponse(status_code=403, content={"licensed": False, "message": LICENSE_MSG})
    return await call_next(request)


# =========================
# 共用小工具
# =========================
def is_image(name):
    return name.lower().endswith(IMAGE_EXTS)


def read_class_names(info):
    """data.yaml 的 names 可能是 dict {0:'A'} 或 list ['A']，統一成 list。"""
    names = (info or {}).get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names or [])


def label_path_for(img_path):
    """Ultralytics 慣例：影像路徑中最後一個 /images → /labels，同副檔名換 .txt。
    路徑內沒有 images/ 時，標籤就與圖片同層（Ultralytics 也是這樣找）。"""
    d, base = os.path.split(img_path)
    stem = os.path.splitext(base)[0]
    sa, sb = os.sep + "images", os.sep + "labels"
    if sa in d:
        head, _, tail = d.rpartition(sa)
        d = head + sb + tail
    return os.path.join(d, stem + ".txt")


def read_labels(img_path, class_names):
    """回傳 [{className,x,y,w,h}]（像素座標，左上原點）；沒有標籤檔就回空陣列。"""
    lp = label_path_for(img_path)
    if not os.path.exists(lp):
        return []
    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception:
        return []
    out = []
    with open(lp, encoding="utf-8") as f:
        for line in f:
            v = line.split()
            if len(v) < 5:
                continue
            try:
                idx = int(float(v[0]))
                name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
                if len(v) == 5:
                    cx, cy, w, h = [float(x) for x in v[1:5]]
                    out.append({"className": name, "x": (cx - w / 2) * W, "y": (cy - h / 2) * H,
                                "w": w * W, "h": h * H})
                else:   # seg / obb：唯讀顯示外接框，輕量版不提供多邊形編輯
                    pts = [float(x) for x in v[1:]]
                    xs, ys = pts[0::2], pts[1::2]
                    x0, x1 = min(xs) * W, max(xs) * W
                    y0, y1 = min(ys) * H, max(ys) * H
                    out.append({"className": name, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                                "readonly": True})
            except (ValueError, IndexError):
                continue
    return out


def write_labels(img_path, boxes, class_names):
    """boxes：[{className,x,y,w,h}] 像素座標 → YOLO 正規化 txt。空清單 = 背景圖（寫出空檔）。"""
    with Image.open(img_path) as im:
        W, H = im.size
    lp = label_path_for(img_path)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    lines = []
    for b in boxes:
        if b.get("readonly"):
            continue
        try:
            idx = class_names.index(b["className"])
        except ValueError:
            continue
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        if nw <= 0 or nh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return lp


# =========================
# 資料集 session（只存路徑，不存圖也不存標註）
# =========================
SESSION = {"open": False, "mode": "", "root": "", "yaml": "", "classes": [],
           "model_type": "detect", "images": [], "warnings": [],
           "sources": [], "orig_yaml": "",
           "split": None, "split_pre_yaml": "", "split_pre_mode": ""}


def dataset_dirs_from_yaml(yaml_path, info, key):
    """解析 data.yaml 的 train/val → 影像資料夾清單。支援 images/train、train/images、
    絕對路徑、list 多目錄，以及只給根目錄（自動往下找 images/）。"""
    entries = info.get(key)
    if not entries:
        return [], []
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        return [], [f"{key}: 格式無法解析"]
    ydir = os.path.dirname(os.path.abspath(yaml_path))
    roots = [ydir]
    if info.get("path"):
        p = str(info["path"])
        roots.insert(0, p if os.path.isabs(p) else os.path.normpath(os.path.join(ydir, p)))
    dirs, problems = [], []
    for e in entries:
        e = str(e)
        cands = [os.path.normpath(e)] if os.path.isabs(e) else \
                [os.path.normpath(os.path.join(r, e)) for r in roots]
        # Roboflow 匯出常寫成 ../train/images，但 train/ 其實就在 data.yaml 旁邊；
        # 直接解析失敗時，把開頭的 ../ 去掉再試一次。
        stripped = e
        while stripped.startswith(("../", "..\\")):
            stripped = stripped[3:]
        if stripped != e and not os.path.isabs(e):
            cands += [os.path.normpath(os.path.join(r, stripped)) for r in roots]
        hit = next((c for c in cands if os.path.isdir(c)), None)
        if hit is None:
            problems.append(f"{key}: 找不到資料夾（{e}）" if not any(os.path.isfile(c) for c in cands)
                            else f"{key}: 不支援圖片清單檔（{e}），請指向資料夾")
            continue
        if stripped != e and hit.endswith(stripped.replace("/", os.sep)):
            problems.append(f"{key}: data.yaml 寫的是「{e}」，實際資料在 {hit}，已自動對應")
        try:
            has_img = any(is_image(f) for f in os.listdir(hit))
        except OSError:
            has_img = False
        if not has_img and os.path.isdir(os.path.join(hit, "images")):
            hit = os.path.join(hit, "images")
        dirs.append(hit)
    return dirs, problems


def scan_images(dirs):
    """多個資料夾的圖片清單（保持順序）。train 與 val 指到同一個資料夾時要去重，
    否則同一張圖會在清單裡出現兩次，資料夾模式拆分時還會同時落進 train 和 val。"""
    out, seen = [], set()
    for d in dirs:
        try:
            for f in sorted(os.listdir(d)):
                if not is_image(f):
                    continue
                p = os.path.join(d, f)
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        except OSError:
            continue
    return out


@app.post("/dataset/open")
async def dataset_open(payload: dict):
    """開啟資料集：{"yaml": "/path/data.yaml"} 或 {"folder": "/path/images"}。
    只掃描檔名，不讀圖片內容，幾萬張也是毫秒等級、記憶體只有路徑字串。"""
    ypath = (payload.get("yaml") or "").strip()
    folder = (payload.get("folder") or "").strip()
    warnings = []

    if ypath:
        if not os.path.isfile(ypath):
            raise HTTPException(404, f"找不到設定檔：{ypath}")
        if os.path.splitext(ypath)[1].lower() not in (".yaml", ".yml"):
            raise HTTPException(400, "請選擇 .yaml / .yml 設定檔")
        try:
            with open(ypath, encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
        except Exception as e:
            raise HTTPException(400, f"data.yaml 解析失敗：{e}")
        if not isinstance(info, dict):
            raise HTTPException(400, "data.yaml 格式錯誤（最外層需為 key: value）")
        classes = read_class_names(info)
        if not classes:
            raise HTTPException(400, "data.yaml 缺少 names（類別清單）")
        dirs = []
        for key in ("train", "val"):
            d, p = dataset_dirs_from_yaml(ypath, info, key)
            dirs += d
            warnings += p
        if not dirs:
            raise HTTPException(400, "train / val 路徑都無法解析，請確認相對路徑的基準位置")
        images = scan_images(dirs)
        raw_type = str(info.get("model_type") or info.get("task") or "").lower()
        model_type = {"segment": "seg", "classify": "cls", "detection": "detect"}.get(raw_type, raw_type) or "detect"
        SESSION.update({"open": True, "mode": "yaml", "root": os.path.dirname(os.path.abspath(ypath)),
                        "yaml": os.path.abspath(ypath), "classes": classes, "model_type": model_type,
                        "images": images, "warnings": warnings})
    elif folder:
        if not os.path.isdir(folder):
            raise HTTPException(404, f"找不到資料夾：{folder}")
        images = scan_images([folder])
        classes = list(payload.get("classes") or [])
        ypath = os.path.join(folder, "data.yaml")
        if not classes and os.path.isfile(ypath):
            try:
                with open(ypath, encoding="utf-8") as f:
                    classes = read_class_names(yaml.safe_load(f) or {})
            except Exception:
                classes = []
        SESSION.update({"open": True, "mode": "folder", "root": os.path.abspath(folder), "yaml": "",
                        "classes": classes, "model_type": "detect", "images": images, "warnings": []})
    else:
        raise HTTPException(400, "請提供 yaml 或 folder")

    SESSION["sources"] = []
    SESSION["orig_yaml"] = ""
    SESSION["split"] = None
    SESSION["split_pre_yaml"] = ""
    SESSION["split_pre_mode"] = ""

    if not SESSION["images"]:
        SESSION["open"] = False
        raise HTTPException(400, "找不到任何圖片（支援 .jpg / .jpeg / .png / .bmp / .webp）")
    log(f"dataset open: mode={SESSION['mode']} root={SESSION['root']} images={len(SESSION['images'])}")
    return dataset_summary()


@app.get("/dataset/summary")
def dataset_summary():
    if not SESSION["open"]:
        return {"open": False}
    labeled = sum(1 for p in SESSION["images"] if os.path.exists(label_path_for(p)))
    return {"open": True, "mode": SESSION["mode"], "root": SESSION["root"], "yaml": SESSION["yaml"],
            "classes": SESSION["classes"], "model_type": SESSION["model_type"],
            "total": len(SESSION["images"]), "labeled": labeled,
            "unlabeled": len(SESSION["images"]) - labeled, "warnings": SESSION["warnings"],
            "sources": SESSION.get("sources", []), "split": SESSION.get("split"),
            "testset": _testset_info(), "trainset": _trainset_info(),
            "consolidated_last": _last_consolidation(),
            # 切到訓練資料夾去訓練之後，前端用這個開回原本的資料集（合併／切分模式重開會失去狀態，不提供）
            "base": ({"yaml": SESSION["yaml"]} if SESSION["mode"] == "yaml" else
                     {"folder": SESSION["root"]} if SESSION["mode"] == "folder" else None)}


@app.get("/dataset/images")
def dataset_images(offset: int = 0, limit: int = 60, filter: str = "all", focus: str = ""):
    """分頁回傳圖片清單。前端一次只拿一頁，DOM 與記憶體都不會爆。
    filter=infer_added 只列「從推論頁加入」的圖；focus=<路徑> 會自動翻到那張圖所在的頁。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    added = {e["path"] for e in _infer_added_mine()} if filter == "infer_added" else set()
    items = []
    members = set(_trainset_manifest(trainset_dir())["items"]) if not _is_trainset_session() else set()
    for p in SESSION["images"]:
        if filter == "infer_added" and p not in added:
            continue
        has = os.path.exists(label_path_for(p))
        in_train = os.path.abspath(p) in members
        if filter == "labeled" and not has:
            continue
        if filter == "unlabeled" and has:
            continue
        if filter == "in_train" and not in_train:
            continue
        if filter == "not_in_train" and in_train:
            continue
        items.append({"path": p, "name": os.path.basename(p), "labeled": has, "in_train": in_train})
    focus_index = -1
    if focus and limit > 0:
        focus_index = next((i for i, it in enumerate(items) if it["path"] == focus), -1)
        if focus_index >= 0:
            offset = focus_index // limit * limit
    return {"total": len(items), "offset": offset, "items": items[offset:offset + limit],
            "focus_index": focus_index - offset if focus_index >= 0 else -1}


@app.get("/dataset/labels")
def dataset_labels(path: str):
    if not os.path.isfile(path):
        raise HTTPException(404, "圖片不存在")
    with Image.open(path) as im:
        W, H = im.size
    return {"path": path, "width": W, "height": H,
            "boxes": read_labels(path, SESSION["classes"]),
            "hasFile": os.path.exists(label_path_for(path))}


def backup_file_for(img_path):
    key = hashlib.md5(os.path.abspath(img_path).encode()).hexdigest()
    return os.path.join(BACKUP_DIR, key + ".json")


UNDO_DEPTH = 10          # 每張圖保留幾層還原點


def read_stack(img_path):
    bf = backup_file_for(img_path)
    if not os.path.exists(bf):
        return []
    try:
        with open(bf, encoding="utf-8") as f:
            return json.load(f).get("stack", [])
    except Exception:
        return []


def write_stack(img_path, stack):
    bf = backup_file_for(img_path)
    if stack:
        with open(bf, "w", encoding="utf-8") as f:
            json.dump({"image": os.path.abspath(img_path), "label_path": label_path_for(img_path),
                       "stack": stack}, f, ensure_ascii=False)
    elif os.path.exists(bf):
        os.remove(bf)


def make_backup(img_path):
    """每次存檔前把「磁碟上現在的標籤」推進還原堆疊，最多留 UNDO_DEPTH 層。
    連續存檔多次也能一層一層退回去。原本沒有標籤檔也要記錄——還原時要把檔案刪掉才算回到原狀。"""
    lp = label_path_for(img_path)
    existed = os.path.exists(lp)
    content = ""
    if existed:
        with open(lp, encoding="utf-8") as f:
            content = f.read()
    entry = {"existed": existed, "content": content,
             "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    stack = read_stack(img_path)
    stack.append(entry)
    write_stack(img_path, stack[-UNDO_DEPTH:])
    return entry


@app.get("/dataset/labels/backup")
def dataset_labels_backup(path: str):
    """這張圖還剩幾層還原點（前端據此決定「復原」按鈕能不能按）。"""
    stack = read_stack(path)
    if not stack:
        return {"has": False, "count": 0}
    return {"has": True, "count": len(stack), "saved_at": stack[-1].get("saved_at", ""),
            "existed": stack[-1].get("existed", False)}


@app.post("/dataset/labels")
async def dataset_labels_save(payload: dict):
    """明確存檔：先備份磁碟上的原始標籤，再覆寫。前端會先跳確認視窗。"""
    path = payload.get("path") or ""
    if not os.path.isfile(path):
        raise HTTPException(404, "圖片不存在")
    rec = make_backup(path)
    lp = write_labels(path, payload.get("boxes") or [], SESSION["classes"])
    _trainset_sync(path)
    if SESSION["mode"] == "folder":
        sync_folder_yaml()
    _infer_added_mark_reviewed(path)
    log(f"labels saved: {lp}（原檔已備份，existed={rec['existed']}）")
    return {"status": "ok", "label_path": lp, "backup_at": rec["saved_at"]}


@app.post("/dataset/labels/restore")
async def dataset_labels_restore(payload: dict):
    """退回上一層還原點。堆疊還有東西就能一直退，所以存檔很多次之後仍然可以復原。"""
    path = payload.get("path") or ""
    stack = read_stack(path)
    if not stack:
        raise HTTPException(404, "這張圖沒有可還原的備份")
    entry = stack.pop()
    lp = label_path_for(path)
    if entry.get("existed"):
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        with open(lp, "w", encoding="utf-8") as f:
            f.write(entry.get("content", ""))
    elif os.path.exists(lp):
        os.remove(lp)
    write_stack(path, stack)
    _trainset_sync(path)
    log(f"labels restored: {lp}（回到 {entry.get('saved_at')} 之前，剩 {len(stack)} 層）")
    return {"status": "ok", "restored_to": entry.get("saved_at", ""), "remaining": len(stack),
            "boxes": read_labels(path, SESSION["classes"]),
            "hasFile": os.path.exists(lp)}


@app.post("/dataset/classes")
async def dataset_classes(payload: dict):
    """類別只允許「新增」與「改名」——刪除或換順序會讓既有 txt 的類別編號全部對錯。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    new = [str(c).strip() for c in (payload.get("classes") or []) if str(c).strip()]
    old = SESSION["classes"]
    if len(new) < len(old):
        raise HTTPException(400, "要刪除類別請用「刪除」按鈕（會一併重編所有標籤的類別編號）")
    SESSION["classes"] = new
    if SESSION["mode"] == "folder":
        sync_folder_yaml()
    elif SESSION["yaml"]:
        sync_yaml_names(SESSION["yaml"], new)
    _trainset_refresh_yaml()
    return {"status": "ok", "classes": new}


@app.get("/dataset/classes/usage")
def class_usage(index: int):
    """刪除前先算清楚會影響多少：幾個框會被刪、幾個檔案要改寫。"""
    boxes = files = 0
    for img in SESSION["images"]:
        lp = label_path_for(img)
        if not os.path.exists(lp):
            continue
        hit = False
        try:
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    v = line.split()
                    if not v:
                        continue
                    try:
                        idx = int(float(v[0]))
                    except ValueError:
                        continue
                    if idx == index:
                        boxes += 1
                        hit = True
                    elif idx > index:
                        hit = True                    # 編號要往前挪，也算要改寫
        except OSError:
            continue
        if hit:
            files += 1
    _trainset_refresh_yaml()
    return {"index": index, "name": SESSION["classes"][index] if index < len(SESSION["classes"]) else "",
            "boxes": boxes, "files": files, "total_images": len(SESSION["images"])}


@app.post("/dataset/classes/delete")
async def class_delete(payload: dict):
    """刪掉一個類別：移除該類別的所有標註框，並把後面類別的編號全部往前挪一號。
    動手前把所有要改的標籤檔整包備份起來，可以一鍵還原。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    index = int(payload.get("index", -1))
    classes = SESSION["classes"]
    if not (0 <= index < len(classes)):
        raise HTTPException(400, "類別編號不正確")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = {"created": stamp, "class": classes[index], "index": index,
              "classes_before": list(classes), "yaml": SESSION.get("yaml", ""),
              "mode": SESSION["mode"], "files": {}}
    removed = touched = 0

    # 測試圖片資料夾的標籤也要一起重編，否則之後在測試資料夾上評估，類別編號會全部對錯
    for img in SESSION["images"] + _testset_images() + _trainset_images():
        lp = label_path_for(img)
        if not os.path.exists(lp):
            continue
        try:
            with open(lp, encoding="utf-8") as f:
                original = f.read()
        except OSError:
            continue
        out, changed = [], False
        for line in original.splitlines():
            v = line.split()
            if not v:
                continue
            try:
                idx = int(float(v[0]))
            except ValueError:
                out.append(line)
                continue
            if idx == index:
                removed += 1
                changed = True
                continue
            if idx > index:
                v[0] = str(idx - 1)
                line = " ".join(v)
                changed = True
            out.append(line)
        if not changed:
            continue
        backup["files"][lp] = original
        with open(lp, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
        touched += 1

    SESSION["classes"] = classes[:index] + classes[index + 1:]
    if SESSION["mode"] == "folder":
        sync_folder_yaml()
    elif SESSION.get("yaml"):
        sync_yaml_names(SESSION["yaml"], SESSION["classes"])

    bpath = os.path.join(BACKUP_DIR, f"classdel_{stamp}.json")
    with open(bpath, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False)
    log(f"class deleted: {backup['class']}（移除 {removed} 個框、改寫 {touched} 個檔案），備份 {bpath}")
    return {"status": "ok", "classes": SESSION["classes"], "removed_boxes": removed,
            "touched_files": touched, "backup": os.path.basename(bpath)}


def _last_classdel():
    files = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("classdel_"))
    return os.path.join(BACKUP_DIR, files[-1]) if files else ""


@app.get("/dataset/classes/deleted")
def class_deleted_info():
    """有沒有可還原的「刪除類別」動作（前端據此顯示還原按鈕）。"""
    p = _last_classdel()
    if not p:
        return {"has": False}
    try:
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
        return {"has": True, "class": b["class"], "created": b["created"], "files": len(b["files"])}
    except Exception:
        return {"has": False}


@app.post("/dataset/classes/restore")
async def class_restore():
    """把最近一次刪除類別的動作整包還原：標籤檔內容與類別清單都回到刪除前。"""
    p = _last_classdel()
    if not p:
        raise HTTPException(404, "沒有可還原的類別刪除紀錄")
    with open(p, encoding="utf-8") as f:
        b = json.load(f)
    for lp, content in b["files"].items():
        try:
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            log(f"還原失敗 {lp}：{e}", "WARNING")
    SESSION["classes"] = b["classes_before"]
    if SESSION["mode"] == "folder":
        sync_folder_yaml()
    elif SESSION.get("yaml"):
        sync_yaml_names(SESSION["yaml"], SESSION["classes"])
    os.remove(p)
    log(f"class delete restored: {b['class']}（{len(b['files'])} 個檔案）")
    _trainset_refresh_yaml()
    return {"status": "ok", "classes": SESSION["classes"], "restored_files": len(b["files"]),
            "class": b["class"]}


def sync_yaml_names(path, classes):
    try:
        with open(path, encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
        info["names"] = {i: n for i, n in enumerate(classes)}
        info["nc"] = len(classes)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(info, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        log(f"寫入 data.yaml 失敗（{path}）：{e}", "WARNING")


def sync_folder_yaml():
    """資料夾模式：在資料夾內維護一份 data.yaml，下次可直接用 yaml 開啟、也能直接餵訓練。"""
    root = SESSION["root"]
    path = os.path.join(root, "data.yaml")
    info = {"path": root, "train": ".", "val": ".", "model_type": "detect",
            "names": {i: n for i, n in enumerate(SESSION["classes"])}, "nc": len(SESSION["classes"])}
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(info, f, allow_unicode=True, sort_keys=False)
        SESSION["yaml"] = path
    except Exception as e:
        log(f"寫入資料夾 data.yaml 失敗：{e}", "WARNING")


# =========================
# 合併外部資料集（整包複製進 .merged/<id>/，原始來源完全不動）
# =========================
MERGE_DIR = os.path.join(OUTPUT_DIR, ".merged")
os.makedirs(MERGE_DIR, exist_ok=True)


def _sanitize_class(name):
    return "".join(c for c in str(name).strip() if c.isalnum() or c in "-_")


def _ext_dataset_info(ypath):
    """讀外部 data.yaml → (info, classes, images, task, split_defined, warnings)。原檔全程唯讀。"""
    if not os.path.isfile(ypath):
        raise HTTPException(404, f"找不到設定檔：{ypath}")
    if os.path.splitext(ypath)[1].lower() not in (".yaml", ".yml"):
        raise HTTPException(400, "請選擇 .yaml / .yml 設定檔")
    try:
        with open(ypath, encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(400, f"data.yaml 解析失敗：{e}")
    if not isinstance(info, dict):
        raise HTTPException(400, "data.yaml 格式錯誤（最外層需為 key: value）")
    classes = read_class_names(info)
    if not classes:
        raise HTTPException(400, "data.yaml 缺少 names（類別清單）")
    dirs, warnings, have = [], [], {"train": False, "val": False}
    for key in ("train", "val"):
        d, p = dataset_dirs_from_yaml(ypath, info, key)
        have[key] = bool(d)
        dirs += d
        warnings += p
    images = scan_images(dirs)
    raw = str(info.get("model_type") or info.get("task") or "").lower()
    task = {"segment": "seg", "classify": "cls", "detection": "detect"}.get(raw, raw) or "detect"
    return info, classes, images, task, (have["train"] and have["val"]), warnings


@app.get("/dataset/merge/scan")
def merge_scan(path: str):
    """讀外部 data.yaml，回報它的類別清單／圖片數／任務型別，給前端建立對應表。"""
    if not SESSION["open"]:
        raise HTTPException(400, "請先載入目前的資料集")
    _info, classes, images, task, split_defined, warnings = _ext_dataset_info(path.strip())
    labeled = sum(1 for p in images if os.path.exists(label_path_for(p)))
    return {"classes": [{"idx": i, "name": n} for i, n in enumerate(classes)],
            "images": len(images), "has_labels": labeled, "task": task,
            "current_task": SESSION["model_type"], "split_defined": split_defined,
            "warnings": warnings}


def _resolve_mapping(mapping, new_names, ext_classes):
    """(remap {ext_idx:int -> sess_idx|None}, working_classes, appended_names, notes)。
    現有類別編號一律不動，只在尾端新增。"""
    working = list(SESSION["classes"])
    lower = {c.lower(): i for i, c in enumerate(working)}
    n_sess = len(SESSION["classes"])
    remap, appended, notes = {}, [], []
    for k in sorted(mapping.keys(), key=lambda x: int(x)):
        ei, v = int(k), mapping[k]
        if v == "__skip__":
            remap[ei] = None
        elif v == "__new__":
            nm = _sanitize_class(new_names.get(str(k)) or (ext_classes[ei] if ei < len(ext_classes) else ""))
            if not nm:
                remap[ei] = None
                notes.append(f"外部類別 {k} 的新名稱無效，已改為略過")
            elif nm.lower() in lower:
                remap[ei] = lower[nm.lower()]
                notes.append(f"「{nm}」已存在，外部類別 {k} 改為對應既有類別 {lower[nm.lower()]}")
            else:
                working.append(nm)
                lower[nm.lower()] = len(working) - 1
                appended.append(nm)
                remap[ei] = len(working) - 1
        else:
            try:
                si = int(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"類別對應格式錯誤：{k} → {v}")
            if not (0 <= si < n_sess):
                raise HTTPException(400, f"類別對應超出範圍：{k} → {v}")
            remap[ei] = si
    return remap, working, appended, notes


def _remap_label_text(text, remap, n_ext):
    """回傳 (out_lines, remapped, dropped, malformed)。只換第 0 欄編號，其餘（含多邊形點）原樣保留。"""
    out, remapped, dropped, malformed = [], 0, 0, 0
    for line in text.splitlines():
        v = line.split()
        if not v:
            continue
        if len(v) < 5:
            malformed += 1
            continue
        try:
            ei = int(float(v[0]))
        except ValueError:
            malformed += 1
            continue
        si = remap.get(ei, "__miss__") if 0 <= ei < n_ext else "__miss__"
        if si == "__miss__" or si is None:
            dropped += 1
            continue
        v[0] = str(si)
        out.append(" ".join(v))
        remapped += 1
    return out, remapped, dropped, malformed


def _class_in_use(index):
    """目前 SESSION 影像的標籤檔裡，還有沒有人用這個類別編號。"""
    for img in SESSION["images"]:
        lp = label_path_for(img)
        if not os.path.exists(lp):
            continue
        try:
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    v = line.split()
                    if not v:
                        continue
                    try:
                        if int(float(v[0])) == index:
                            return True
                    except ValueError:
                        continue
        except OSError:
            continue
    return False


def rebuild_merged_yaml():
    """依 SESSION["sources"] 產生 .merged/_active/data.yaml；沒有來源就還原回原始 yaml。"""
    sources = SESSION.get("sources") or []
    if not sources:
        if SESSION.get("orig_yaml"):
            SESSION["yaml"], SESSION["mode"] = SESSION["orig_yaml"], "yaml"
        return SESSION["yaml"]
    orig = SESSION.get("orig_yaml") or SESSION["yaml"]
    try:
        with open(orig, encoding="utf-8") as f:
            oinfo = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(400, f"讀取原始 data.yaml 失敗：{e}")
    train_dirs, _ = dataset_dirs_from_yaml(orig, oinfo, "train")
    val_dirs, _ = dataset_dirs_from_yaml(orig, oinfo, "val")
    train_dirs, val_dirs = list(train_dirs), list(val_dirs)
    for s in sources:
        idir = os.path.join(s["stage_root"], "images")
        train_dirs.append(idir)
        if s.get("split") == "train_val":
            val_dirs.append(idir)
    if not val_dirs:
        val_dirs = list(train_dirs)
    out_dir = os.path.join(MERGE_DIR, "_active")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump({"path": out_dir,
                   "train": [os.path.abspath(d) for d in train_dirs],
                   "val": [os.path.abspath(d) for d in val_dirs],
                   "names": {i: n for i, n in enumerate(SESSION["classes"])},
                   "nc": len(SESSION["classes"])}, f, allow_unicode=True, sort_keys=False)
    SESSION["yaml"], SESSION["mode"] = path, "merged"
    return path


@app.post("/dataset/merge")
async def dataset_merge(payload: dict):
    """把外部資料集的影像整包複製進 .merged/<id>/images/，標籤依對應表重新編號寫進 labels/。
    dry_run=True 只回報計畫、不寫任何檔案。原始來源與其 data.yaml 全程不動。"""
    if not SESSION["open"]:
        raise HTTPException(400, "請先載入目前的資料集")
    if SESSION["mode"] not in ("yaml", "merged"):
        raise HTTPException(400, "合併功能目前僅支援以 data.yaml 載入的資料集")
    if TRAIN["state"] == "running":
        raise HTTPException(409, "訓練進行中，請結束後再合併")

    ypath = (payload.get("yaml") or "").strip()
    mapping = payload.get("mapping") or {}
    new_names = {str(k): v for k, v in (payload.get("new_names") or {}).items()}
    if not ypath:
        raise HTTPException(400, "請提供外部 data.yaml 路徑")
    if not mapping:
        raise HTTPException(400, "請先設定類別對應")
    split = payload.get("split") if payload.get("split") in ("train", "train_val") else "train"
    include_unlabeled = bool(payload.get("include_unlabeled"))
    dedup = "rename" if payload.get("dedup") == "rename" else "skip"
    dry_run = bool(payload.get("dry_run"))

    _info, ext_classes, ext_images, ext_task, _sd, warnings = _ext_dataset_info(ypath)
    if ext_task and SESSION["model_type"] and ext_task != SESSION["model_type"]:
        raise HTTPException(400, f"任務型別不符：外部為 {ext_task}，目前資料集為 {SESSION['model_type']}")

    remap, working, appended, notes = _resolve_mapping(mapping, new_names, ext_classes)
    n_ext = len(ext_classes)
    seen = {os.path.basename(p) for p in SESSION["images"]}

    plan = {"images_added": 0, "labels_written": 0, "boxes_remapped": 0, "boxes_dropped": 0,
            "files_malformed": 0, "no_label": 0, "duplicate": 0, "classes_appended": appended,
            "class_count": [len(SESSION["classes"]), len(working)], "notes": notes + warnings}

    base_id = (_sanitize_class(os.path.basename(os.path.dirname(os.path.abspath(ypath)))) or "src") \
        + datetime.now().strftime("_%Y%m%d_%H%M%S")
    src_id, _n = base_id, 1
    while os.path.exists(os.path.join(MERGE_DIR, src_id)):
        src_id = f"{base_id}_{_n}"
        _n += 1
    stage_root = os.path.join(MERGE_DIR, src_id)
    img_out, lbl_out = os.path.join(stage_root, "images"), os.path.join(stage_root, "labels")
    if not dry_run:
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

    used, staged = set(), []
    for img in ext_images:
        base = os.path.basename(img)
        lp = label_path_for(img)
        has_label = os.path.exists(lp)
        if not has_label and not include_unlabeled:
            plan["no_label"] += 1
            continue
        if dedup == "skip" and base in seen:
            plan["duplicate"] += 1
            continue
        stem, ext = os.path.splitext(f"{src_id}__{base}" if dedup == "rename" else base)
        name, i = stem + ext, 1
        while name in used:
            name = f"{stem}_{i}{ext}"
            i += 1
        used.add(name)

        lines, rm, dr, mal = [], 0, 0, 0
        if has_label:
            try:
                with open(lp, encoding="utf-8") as f:
                    lines, rm, dr, mal = _remap_label_text(f.read(), remap, n_ext)
            except OSError:
                mal += 1
        plan["boxes_remapped"] += rm
        plan["boxes_dropped"] += dr
        plan["files_malformed"] += mal

        if not dry_run:
            try:
                shutil.copy2(img, os.path.join(img_out, name))
            except OSError as e:
                plan["notes"].append(f"複製失敗 {base}：{e}")
                continue
            with open(os.path.join(lbl_out, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            staged.append(os.path.join(img_out, name))

        plan["images_added"] += 1
        if has_label:
            plan["labels_written"] += 1

    if dry_run:
        return {"dry_run": True, "plan": plan}

    if plan["images_added"] == 0:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise HTTPException(400, "沒有任何影像會被匯入（可能全部重複或都沒有標籤檔）")

    if not SESSION.get("orig_yaml"):
        SESSION["orig_yaml"] = SESSION["yaml"]
    SESSION["classes"] = working
    SESSION["images"] = SESSION["images"] + staged
    entry = {"source_id": src_id, "origin": os.path.abspath(ypath), "stage_root": stage_root,
             "images_added": plan["images_added"], "classes_added": appended, "split": split,
             "merged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    SESSION.setdefault("sources", []).append(entry)
    try:
        with open(os.path.join(stage_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(dict(entry, mapping=mapping, new_names=new_names, plan=plan), f, ensure_ascii=False)
    except OSError:
        pass
    plan["merged_yaml"] = rebuild_merged_yaml()
    log(f"merge: +{src_id} images={plan['images_added']} appended={appended} "
        f"remapped={plan['boxes_remapped']} dropped={plan['boxes_dropped']}")
    return {"dry_run": False, "plan": plan, "source_id": src_id, "summary": dataset_summary()}


@app.post("/dataset/merge/remove")
async def dataset_merge_remove(payload: dict):
    """移除一個已合併來源：刪掉 .merged/<id>/（複製的影像＋重編號標籤），
    盡量回收只由這次合併新增、且沒有其他人使用的尾端類別。原始來源不受影響。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    if TRAIN["state"] == "running":
        raise HTTPException(409, "訓練進行中，請結束後再移除")
    sid = (payload.get("source_id") or "").strip()
    sources = SESSION.get("sources") or []
    entry = next((s for s in sources if s["source_id"] == sid), None)
    if not entry:
        raise HTTPException(404, f"找不到已合併來源：{sid}")

    stage_root = entry.get("stage_root") or os.path.join(MERGE_DIR, sid)
    prefix = os.path.abspath(os.path.join(stage_root, "images")) + os.sep
    SESSION["images"] = [p for p in SESSION["images"] if not os.path.abspath(p).startswith(prefix)]
    SESSION["sources"] = [s for s in sources if s["source_id"] != sid]

    other_added = {c for s in SESSION["sources"] for c in (s.get("classes_added") or [])}
    removed, kept = [], []
    for name in reversed(list(entry.get("classes_added") or [])):
        if name not in SESSION["classes"]:
            continue
        idx = SESSION["classes"].index(name)
        if idx != len(SESSION["classes"]) - 1:
            kept.append({"name": name, "reason": "後面已有其他類別，移除會打亂既有標籤編號"})
        elif name in other_added:
            kept.append({"name": name, "reason": "其他已合併來源也在使用"})
        elif _class_in_use(idx):
            kept.append({"name": name, "reason": "仍有標籤檔使用此類別"})
        else:
            SESSION["classes"].pop(idx)
            removed.append(name)

    shutil.rmtree(stage_root, ignore_errors=True)
    merged_yaml = rebuild_merged_yaml()
    log(f"merge removed: {sid} rolled_back={removed} kept={[k['name'] for k in kept]}")
    return {"status": "ok", "removed_classes": removed, "kept_classes": kept,
            "merged_yaml": merged_yaml, "summary": dataset_summary()}


CONSOLIDATED_DIR = os.path.join(MERGE_DIR, ".consolidated")


def _consolidation_logs():
    """目前資料集（依 root）還沒被復原的「併入」紀錄，舊→新。"""
    try:
        names = sorted(f for f in os.listdir(CONSOLIDATED_DIR) if f.endswith(".json"))
    except OSError:
        return []
    out = []
    for f in names:
        try:
            with open(os.path.join(CONSOLIDATED_DIR, f), encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("root") == SESSION.get("root") and not rec.get("undone"):
            rec["_file"] = os.path.join(CONSOLIDATED_DIR, f)
            out.append(rec)
    return out


def _last_consolidation():
    logs = _consolidation_logs() if SESSION["open"] else []
    if not logs:
        return None
    r = logs[-1]
    return {"source_id": r["source_id"], "count": len(r["files"]), "target": r["target"], "at": r["at"]}


@app.post("/dataset/merge/consolidate")
def dataset_merge_consolidate(payload: dict):
    """把一個已合併來源「真的」複製進原始資料集的 train 資料夾（圖片＋已重新編號的標籤），
    原始 data.yaml 的 names 補上合併時新增的類別，然後拿掉這個合併來源與它在 .merged/ 的暫存複本。
    之後重開原始資料集也看得到這些圖。複製了哪些檔案會記錄下來，可用「復原上次併入」刪掉。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    if SESSION.get("split"):
        raise HTTPException(400, "目前套用了訓練／驗證／測試切分，請先撤銷切分再併入")
    if _training_uses(SESSION["root"]):
        raise HTTPException(409, "這個資料集正在訓練中，請等訓練結束再併入")
    sid = (payload.get("source_id") or "").strip()
    entry = next((s for s in SESSION.get("sources") or [] if s["source_id"] == sid), None)
    if not entry:
        raise HTTPException(404, f"找不到已合併來源：{sid}")
    orig = SESSION.get("orig_yaml") or ""
    try:
        with open(orig, encoding="utf-8") as f:
            oinfo = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(400, f"讀取原始 data.yaml 失敗：{e}")
    train_dirs, _ = dataset_dirs_from_yaml(orig, oinfo, "train")
    if not train_dirs:
        raise HTTPException(400, "原始 data.yaml 找不到 train 資料夾")
    target = train_dirs[0]

    stage_root = entry.get("stage_root") or os.path.join(MERGE_DIR, sid)
    prefix = os.path.abspath(os.path.join(stage_root, "images")) + os.sep
    staged = [p for p in SESSION["images"] if os.path.abspath(p).startswith(prefix)]
    lbl_dir = os.path.dirname(label_path_for(os.path.join(target, "x.jpg")))
    os.makedirs(lbl_dir, exist_ok=True)

    ts_dir = trainset_dir()
    ts = _trainset_manifest(ts_dir)
    moved, files, renamed = {}, [], False
    for p in staged:
        name = _free_name(target, lbl_dir, os.path.basename(p))
        dest = os.path.join(target, name)
        shutil.copy2(p, dest)
        rec = {"image": dest, "label": ""}
        src_lbl = label_path_for(p)
        if os.path.isfile(src_lbl):
            shutil.copy2(src_lbl, label_path_for(dest))
            rec["label"] = label_path_for(dest)
        files.append(rec)
        moved[p] = dest
        it = ts["items"].pop(os.path.abspath(p), None)        # 已在訓練資料夾裡的，改記到新路徑
        if it:
            ts["items"][os.path.abspath(dest)] = it
            renamed = True
    if renamed:
        _write_trainset_manifest(ts_dir, ts)

    SESSION["images"] = [moved.get(p, p) for p in SESSION["images"]]
    sync_yaml_names(orig, SESSION["classes"])                 # 標籤用到新增的類別編號，原始 yaml 要跟著補上
    os.makedirs(CONSOLIDATED_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(CONSOLIDATED_DIR, f"{stamp}_{sid}.json"), "w", encoding="utf-8") as f:
        json.dump({"source_id": sid, "origin": entry.get("origin"), "root": SESSION["root"], "orig_yaml": orig,
                   "target": target, "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "classes": SESSION["classes"], "files": files}, f, ensure_ascii=False, indent=1)
    SESSION["sources"] = [s for s in SESSION["sources"] if s["source_id"] != sid]
    shutil.rmtree(stage_root, ignore_errors=True)
    rebuild_merged_yaml()
    log(f"merge consolidated: {sid} -> {target} files={len(files)}")
    return {"status": "ok", "copied": len(files), "target": target, "summary": dataset_summary()}


@app.post("/dataset/merge/consolidate/undo")
def dataset_merge_consolidate_undo():
    """復原最近一次「併入目前資料夾」：刪掉當時複製進來的圖片與標籤（只刪紀錄裡的那些檔案）。
    合併來源不會自動接回來，需要的話重新合併一次即可（外部原始資料夾沒有被動過）。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    if _training_uses(SESSION["root"]):
        raise HTTPException(409, "這個資料集正在訓練中，請等訓練結束再復原")
    logs = _consolidation_logs()
    if not logs:
        raise HTTPException(404, "沒有可復原的併入紀錄")
    rec = logs[-1]
    gone = set()
    for f in rec["files"]:
        for p in (f.get("image"), f.get("label")):
            if p and os.path.isfile(p):
                os.remove(p)
        gone.add(os.path.abspath(f["image"]))
        _trainset_drop(f["image"])
    SESSION["images"] = [p for p in SESSION["images"] if os.path.abspath(p) not in gone]
    rec["undone"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = rec.pop("_file")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    log(f"merge consolidate undone: {rec['source_id']} removed={len(rec['files'])}")
    return {"status": "ok", "removed": len(rec["files"]), "source_id": rec["source_id"],
            "summary": dataset_summary()}


# =========================
# 標註頁：移除圖片（連同標籤搬到 .removed_images/，不進訓練，可復原；支援單張與整批）
# =========================
REMOVED_DIR = os.path.join(OUTPUT_DIR, ".removed_images")
os.makedirs(REMOVED_DIR, exist_ok=True)
_REMOVED_LOG = os.path.join(REMOVED_DIR, "log.json")


def _removed_log():
    try:
        with open(_REMOVED_LOG, encoding="utf-8") as f:
            return json.load(f) or []
    except (OSError, json.JSONDecodeError):
        return []


def _write_removed_log(stack):
    with open(_REMOVED_LOG, "w", encoding="utf-8") as f:
        json.dump(stack, f, ensure_ascii=False)


def _reinsert_image(path):
    """把復原的圖片放回 SESSION['images'] 中同資料夾、依檔名排序的位置。"""
    d = os.path.dirname(path)
    imgs = SESSION["images"]
    for i, p in enumerate(imgs):
        if os.path.dirname(p) == d and p > path:
            imgs.insert(i, path)
            return
    same = [i for i, p in enumerate(imgs) if os.path.dirname(p) == d]
    imgs.insert(same[-1] + 1, path) if same else imgs.append(path)


def _remove_one_image(match, batch_id=None):
    """把一張圖片＋標籤搬到 .removed_images/，回傳要寫進 log 的 entry；圖片已經不在磁碟就回傳 None。
    batch_id 有值時，同一批移除的所有 entry 共用同一個 id，之後可以一次全部復原。"""
    _trainset_drop(match)                 # 原圖不要了，訓練資料夾裡的複本也一起拿掉
    if not os.path.isfile(match):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    lp = label_path_for(match)
    img_trash = os.path.join(REMOVED_DIR, f"{stamp}_{os.path.basename(match)}")
    shutil.move(match, img_trash)
    entry = {"stamp": stamp, "root": SESSION.get("root"), "batch_id": batch_id,
             "image_orig": os.path.abspath(match), "image_trash": img_trash,
             "label_orig": "", "label_trash": ""}
    if os.path.isfile(lp):
        lbl_trash = os.path.join(REMOVED_DIR, f"{stamp}_{os.path.basename(lp)}")
        shutil.move(lp, lbl_trash)
        entry["label_orig"] = os.path.abspath(lp)
        entry["label_trash"] = lbl_trash
    return entry


def _restore_one(stack, idx):
    """復原 stack[idx] 這一筆並從 stack 移除；回傳 (ok, name_or_error)。不寫檔，呼叫端自己 _write_removed_log。"""
    entry = stack[idx]
    dest = entry["image_orig"]
    if os.path.exists(dest):
        return False, f"{os.path.basename(dest)}（原位置已有同名檔案）"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(entry["image_trash"], dest)
    except OSError as e:
        return False, f"{os.path.basename(dest)}（{e}）"
    if entry.get("label_trash") and os.path.isfile(entry["label_trash"]) and not os.path.exists(entry["label_orig"]):
        os.makedirs(os.path.dirname(entry["label_orig"]), exist_ok=True)
        shutil.move(entry["label_trash"], entry["label_orig"])
    if not any(p == dest or os.path.abspath(p) == dest for p in SESSION["images"]):
        _reinsert_image(dest)
    stack.pop(idx)
    return True, os.path.basename(dest)


@app.get("/dataset/image/removed")
def dataset_image_removed():
    """目前資料集有沒有可復原的已移除圖片；last_batch_count 讓前端知道最近一次是不是整批移除
    （>1 就顯示「復原整批」，否則跟以前一樣顯示「復原移除的圖片」）。"""
    mine = [e for e in _removed_log() if e.get("root") == SESSION.get("root")]
    if not mine:
        return {"has": False, "count": 0, "last_batch_count": 0}
    last_batch = mine[-1].get("batch_id")
    n = 1
    if last_batch:
        n = 0
        for e in reversed(mine):
            if e.get("batch_id") != last_batch:
                break
            n += 1
    return {"has": True, "count": len(mine), "last": os.path.basename(mine[-1]["image_orig"]),
            "last_batch_count": n}


@app.post("/dataset/image/remove")
async def dataset_image_remove(payload: dict):
    """把一張圖片與它的標籤檔搬到 .removed_images/。訓練時掃資料夾就不會再看到它。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    raw = (payload.get("path") or "").strip()
    match = next((p for p in SESSION["images"]
                  if p == raw or os.path.abspath(p) == os.path.abspath(raw)), None)
    if not match:
        raise HTTPException(404, "這張圖片不在目前的資料集")
    _guard_image_removal([match])
    SESSION["images"] = [p for p in SESSION["images"] if p != match]

    entry = _remove_one_image(match)
    if entry is None:
        log(f"image remove: {match} 已不在磁碟，僅從清單移除")
        return {"status": "ok", "removed": os.path.basename(match), "already_missing": True,
                "count": len(_removed_log()), "summary": dataset_summary()}

    stack = _removed_log()
    stack.append(entry)
    _write_removed_log(stack)
    log(f"image removed: {match} -> {entry['image_trash']}")
    return {"status": "ok", "removed": os.path.basename(match),
            "count": len([e for e in stack if e.get('root') == SESSION.get('root')]),
            "summary": dataset_summary()}


@app.post("/dataset/image/remove_bulk")
async def dataset_image_remove_bulk(payload: dict):
    """一次移除多張圖片（連同標籤），同一批次共用一個 batch_id，方便一次全部復原。
    找不到的路徑會跳過並回報，不會讓整個請求失敗。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    raws = payload.get("paths") or []
    if not raws:
        raise HTTPException(400, "沒有指定要移除的圖片")
    raw_set = set(raws)
    abs_set = {os.path.abspath(p) for p in raws}
    matches = [p for p in SESSION["images"] if p in raw_set or os.path.abspath(p) in abs_set]
    if not matches:
        raise HTTPException(404, "選取的圖片都不在目前的資料集")
    _guard_image_removal(matches)

    batch_id = datetime.now().strftime("b%Y%m%d_%H%M%S_%f")
    stack = _removed_log()
    removed, skipped = [], []
    for match in matches:
        entry = _remove_one_image(match, batch_id)
        if entry is None:
            skipped.append(os.path.basename(match))
            continue
        stack.append(entry)
        removed.append(os.path.basename(match))
    _write_removed_log(stack)
    matched_set = set(matches)
    SESSION["images"] = [p for p in SESSION["images"] if p not in matched_set]

    log(f"bulk image remove: batch={batch_id} removed={len(removed)} skipped={len(skipped)}")
    return {"status": "ok", "removed": len(removed), "skipped": skipped,
            "count": len([e for e in stack if e.get('root') == SESSION.get('root')]),
            "summary": dataset_summary()}


@app.post("/dataset/image/restore")
async def dataset_image_restore(payload: dict):
    """把最近一張（屬於目前資料集的）已移除圖片搬回原位，標籤檔一起還原——不論它是不是整批移除的一部分。"""
    stack = _removed_log()
    mine = [i for i, e in enumerate(stack) if e.get("root") == SESSION.get("root")]
    if not mine:
        raise HTTPException(404, "沒有可復原的已移除圖片")
    ok, msg = _restore_one(stack, mine[-1])
    _write_removed_log(stack)
    if not ok:
        raise HTTPException(409, f"原位置已有同名檔案，未復原：{msg}")
    log(f"image restored: {msg}")
    return {"status": "ok", "name": msg,
            "count": len([e for e in stack if e.get('root') == SESSION.get('root')]),
            "summary": dataset_summary()}


@app.post("/dataset/image/restore_batch")
async def dataset_image_restore_batch():
    """把最近一次「整批移除」的所有圖片一次復原；如果最近一筆不是整批移除的，效果等同單張復原。"""
    stack = _removed_log()
    mine_idx = [i for i, e in enumerate(stack) if e.get("root") == SESSION.get("root")]
    if not mine_idx:
        raise HTTPException(404, "沒有可復原的已移除圖片")
    last_batch = stack[mine_idx[-1]].get("batch_id")
    targets = []
    for i in reversed(mine_idx):
        e = stack[i]
        if last_batch and e.get("batch_id") == last_batch:
            targets.append(i)
        elif not last_batch and i == mine_idx[-1]:
            targets.append(i)
        else:
            break

    restored, failed = [], []
    for i in sorted(targets, reverse=True):     # 先復原索引大的，pop 才不會弄亂還沒處理的索引
        ok, msg = _restore_one(stack, i)
        (restored if ok else failed).append(msg)
    _write_removed_log(stack)
    log(f"batch restore: restored={len(restored)} failed={len(failed)}")
    return {"status": "ok", "restored": len(restored), "failed": failed,
            "count": len([e for e in stack if e.get('root') == SESSION.get('root')]),
            "summary": dataset_summary()}


# =========================
# 測試圖片資料夾：從訓練資料挑出來、實體搬離的 held-out 圖片
# =========================
TESTSET_DIR = os.path.join(OUTPUT_DIR, "test_images")


def _per_dataset_dir(base, root=None):
    """<base>/<資料集名>_<路徑雜湊>/：每個資料集一個專屬資料夾。名稱加上路徑雜湊，
    兩個同名資料夾的資料集不會共用同一個。"""
    root = os.path.abspath(root or SESSION.get("root") or "")
    name = os.path.basename(root.rstrip(os.sep)) or "dataset"
    return os.path.join(base, f"{name}_{hashlib.sha1(root.encode()).hexdigest()[:6]}")


def testset_dir(root=None):
    """目前資料集專屬的測試圖片資料夾：DefectMaster_Output/test_images/<資料集名>_<路徑雜湊>/。
    放在任何資料集資料夾之外，開資料集、訓練掃描都碰不到；不加點開頭，檔案瀏覽器與推論頁找得到。"""
    return _per_dataset_dir(TESTSET_DIR, root)


def _testset_images(d=None):
    img_dir = os.path.join(d or testset_dir(), "images")
    try:
        return sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if is_image(f))
    except OSError:
        return []


def _testset_log(d):
    try:
        with open(os.path.join(d, "moves.json"), encoding="utf-8") as f:
            return json.load(f) or []
    except (OSError, json.JSONDecodeError):
        return []


def _write_testset_log(d, stack):
    with open(os.path.join(d, "moves.json"), "w", encoding="utf-8") as f:
        json.dump(stack, f, ensure_ascii=False, indent=1)


def _testset_info():
    if not SESSION["open"]:
        return {"dir": "", "count": 0, "labeled": 0, "last_batch_count": 0}
    d = testset_dir()
    imgs = _testset_images(d)
    stack = _testset_log(d)
    n_last = 0
    if stack:
        last = stack[-1].get("batch_id")
        for e in reversed(stack):
            if e.get("batch_id") != last:
                break
            n_last += 1
    return {"dir": d, "count": len(imgs), "labeled": sum(1 for p in imgs if os.path.exists(label_path_for(p))),
            "last_batch_count": n_last}


def _write_testset_yaml(d):
    """產生給 model.val(split="test") 用的 data.yaml + test.txt，回傳 (yaml 路徑, 已標註張數)。
    test.txt 只列「有標籤檔」的圖：沒標註的圖混進去會被當成「沒有任何目標」的背景圖計分，拉低 mAP。
    train / val 是 Ultralytics 規定必填的欄位，只好也指向同一份清單——這個資料夾只拿來評估，不拿來訓練。"""
    labeled = [p for p in _testset_images(d) if os.path.exists(label_path_for(p))]
    tl = os.path.join(d, "test.txt")
    with open(tl, "w", encoding="utf-8") as f:
        f.write("\n".join(labeled) + ("\n" if labeled else ""))
    info = {"path": d, "train": tl, "val": tl, "test": tl,
            "names": {i: n for i, n in enumerate(SESSION["classes"])}, "nc": len(SESSION["classes"])}
    yp = os.path.join(d, "data.yaml")
    with open(yp, "w", encoding="utf-8") as f:
        f.write("# DefectMaster Lite 測試圖片資料夾：只用來評估模型，請勿拿來訓練\n")
        yaml.dump(info, f, allow_unicode=True, sort_keys=False)
    return yp, len(labeled)


def _free_name(img_dir, lbl_dir, base):
    """不同來源資料夾可能有同名圖片；撞名就加 _1、_2…，連標籤檔名（同 stem 不同副檔名也算）一起避開。"""
    stem, ext = os.path.splitext(base)
    cand, k = base, 1
    while (os.path.exists(os.path.join(img_dir, cand)) or
           os.path.exists(os.path.join(lbl_dir, os.path.splitext(cand)[0] + ".txt"))):
        cand, k = f"{stem}_{k}{ext}", k + 1
    return cand


def _split_lists_edit(remove=(), add=None):
    """同步目前套用中的切分清單（train/val/test.txt）：remove 的路徑從清單拿掉，add={'train': [...]} 放回去。
    回傳 {絕對路徑: 原本所在的清單名}，移回時才能放回同一份清單。沒有套用切分就什麼都不做。"""
    split = SESSION.get("split")
    if not split:
        return {}
    split_dir = os.path.dirname(split["yaml"])
    rm = {os.path.abspath(p) for p in remove}
    add = add or {}
    where = {}
    for name in ("train", "val", "test"):
        fp = os.path.join(split_dir, f"{name}.txt")
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        keep = []
        for p in items:
            if os.path.abspath(p) in rm:
                where[os.path.abspath(p)] = name
            else:
                keep.append(p)
        back = [p for p in add.get(name, []) if p not in keep]
        dropped = len(items) - len(keep)
        if not dropped and not back:
            continue
        keep += back
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        counts = split.setdefault("counts", {})
        counts[name] = max(0, counts.get(name, 0) - dropped + len(back))
    return where


@app.get("/dataset/testset")
def dataset_testset():
    return _testset_info()


@app.post("/dataset/testset/move")
def dataset_testset_move(payload: dict):
    """把選取的圖片（連同標籤檔）搬到這個資料集的測試圖片資料夾——不是刪除，是移出訓練資料。
    同一批共用 batch_id，可一次移回。同步 def：大量搬檔時在執行緒池跑，不卡住其他請求。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    raws = payload.get("paths") or []
    if not raws:
        raise HTTPException(400, "沒有指定要移動的圖片")
    raw_set = set(raws)
    abs_set = {os.path.abspath(p) for p in raws}
    matches = [p for p in SESSION["images"] if p in raw_set or os.path.abspath(p) in abs_set]
    if not matches:
        raise HTTPException(404, "選取的圖片都不在目前的資料集")
    _guard_image_removal(matches)

    d = testset_dir()
    img_dir, lbl_dir = os.path.join(d, "images"), os.path.join(d, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    batch_id = datetime.now().strftime("t%Y%m%d_%H%M%S_%f")
    where = _split_lists_edit(remove=matches)
    stack = _testset_log(d)
    moved, skipped = 0, []
    for match in matches:
        _trainset_drop(match)             # 測試圖若還留在訓練資料夾裡，測試就失去意義
        if not os.path.isfile(match):
            skipped.append(os.path.basename(match))
            continue
        name = _free_name(img_dir, lbl_dir, os.path.basename(match))
        dest = os.path.join(img_dir, name)
        lp = label_path_for(match)
        shutil.move(match, dest)
        entry = {"batch_id": batch_id, "stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "root": SESSION.get("root"), "image_orig": os.path.abspath(match), "image_dest": dest,
                 "label_orig": "", "label_dest": "", "split_list": where.get(os.path.abspath(match), "")}
        if os.path.isfile(lp):
            ld = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            shutil.move(lp, ld)
            entry["label_orig"], entry["label_dest"] = os.path.abspath(lp), ld
        stack.append(entry)
        moved += 1
    _write_testset_log(d, stack)
    matched_set = set(matches)
    SESSION["images"] = [p for p in SESSION["images"] if p not in matched_set]
    _write_testset_yaml(d)
    log(f"move to testset: batch={batch_id} moved={moved} skipped={len(skipped)} -> {d}")
    return {"status": "ok", "moved": moved, "skipped": skipped, "testset": _testset_info(),
            "summary": dataset_summary()}


@app.post("/dataset/testset/restore")
def dataset_testset_restore():
    """把最近一批移到測試資料夾的圖片搬回原位（標籤、切分清單一併放回）。
    原位置已有同名檔案的會留在測試資料夾並回報，不會覆蓋。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    d = testset_dir()
    stack = _testset_log(d)
    if not stack:
        raise HTTPException(404, "沒有可移回的測試圖片")
    last = stack[-1].get("batch_id")
    targets = []
    for i in range(len(stack) - 1, -1, -1):
        if stack[i].get("batch_id") != last:
            break
        targets.append(i)

    restored, failed, back = [], [], {}
    for i in targets:                     # 由後往前 pop，還沒處理的索引不會亂掉
        e = stack[i]
        dest = e["image_orig"]
        if not os.path.isfile(e["image_dest"]):
            stack.pop(i)                  # 使用者已經自己把它從測試資料夾拿走了
            failed.append(f"{os.path.basename(dest)}（測試資料夾裡已經沒有這張）")
            continue
        if os.path.exists(dest):
            failed.append(f"{os.path.basename(dest)}（原位置已有同名檔案）")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(e["image_dest"], dest)
        if e.get("label_dest") and os.path.isfile(e["label_dest"]) and not os.path.exists(e["label_orig"]):
            os.makedirs(os.path.dirname(e["label_orig"]), exist_ok=True)
            shutil.move(e["label_dest"], e["label_orig"])
        if not any(p == dest or os.path.abspath(p) == dest for p in SESSION["images"]):
            _reinsert_image(dest)
        if e.get("split_list"):
            back.setdefault(e["split_list"], []).append(dest)
        stack.pop(i)
        restored.append(os.path.basename(dest))
    _split_lists_edit(add=back)
    _write_testset_log(d, stack)
    _write_testset_yaml(d)
    log(f"testset restore: batch={last} restored={len(restored)} failed={len(failed)}")
    return {"status": "ok", "restored": len(restored), "failed": failed, "testset": _testset_info(),
            "summary": dataset_summary()}


# =========================
# 訓練資料夾：把修正好的圖片＋標籤「複製」成一個獨立的 YOLO 資料集，先拿去訓練一版
# =========================
TRAINSET_DIR = os.path.join(OUTPUT_DIR, "train_sets")


def trainset_dir(root=None):
    """目前資料集專屬的訓練資料夾 DefectMaster_Output/train_sets/<資料集名>_<路徑雜湊>/。
    同一個資料夾持續累積：改完 1–200 張先訓練一版，之後繼續加，再訓練下一版。"""
    return _per_dataset_dir(TRAINSET_DIR, root)


def _is_trainset_session():
    """目前開著的資料集本身就是某個訓練資料夾（為了訓練而開啟它）。"""
    return bool(SESSION.get("root")) and \
        os.path.abspath(SESSION["root"]).startswith(os.path.abspath(TRAINSET_DIR) + os.sep)


def _trainset_manifest(d):
    """manifest.json：{"items": {原圖絕對路徑: {"split": "train"|"val", "name": 複本檔名, ...}}}"""
    try:
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        m = {}
    m.setdefault("items", {})
    return m


def _write_trainset_manifest(d, m):
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def _trainset_paths(d, it):
    img = os.path.join(d, "images", it["split"], it["name"])
    return img, os.path.join(d, "labels", it["split"], os.path.splitext(it["name"])[0] + ".txt")


def _trainset_images():
    if not SESSION["open"] or _is_trainset_session():
        return []
    d = trainset_dir()
    return [_trainset_paths(d, it)[0] for it in _trainset_manifest(d)["items"].values()]


def _atomic_copy(src, dst):
    """先寫到同資料夾的暫存檔再 os.replace：訓練就算正在讀這個資料夾，也只會讀到舊檔或新檔，
    不會讀到寫一半的檔。暫存檔副檔名是 .tmp，不會被當成圖片掃進去。"""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _pick_trainset_split(m, name):
    """約 10% 進 val，依檔名雜湊固定分配（同一張圖不會在 train / val 間跳來跳去）。
    train、val 各至少要有一張，Ultralytics 才能訓練——前兩張先各補一張。"""
    splits = {it["split"] for it in m["items"].values()}
    if "train" not in splits:
        return "train"
    if "val" not in splits:
        return "val"
    return "val" if int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def _write_trainset_yaml(d):
    info = {"path": d, "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(SESSION["classes"])}, "nc": len(SESSION["classes"])}
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"# DefectMaster Lite 訓練資料夾：從 {SESSION.get('root')} 挑選修正好的圖片複製而來\n")
        yaml.dump(info, f, allow_unicode=True, sort_keys=False)


def _trainset_refresh_yaml():
    """類別改名／新增／刪除後，讓訓練資料夾的 data.yaml 跟著更新。"""
    if SESSION["open"] and not _is_trainset_session() and os.path.isdir(trainset_dir()):
        _write_trainset_yaml(trainset_dir())


def _trainset_info():
    if not SESSION["open"]:
        return {"dir": "", "yaml": "", "count": 0, "train": 0, "val": 0, "is_trainset": False}
    if _is_trainset_session():
        return {"dir": SESSION["root"], "yaml": SESSION["yaml"], "count": len(SESSION["images"]),
                "train": 0, "val": 0, "is_trainset": True}
    d = trainset_dir()
    items = _trainset_manifest(d)["items"].values()
    n_val = sum(1 for it in items if it["split"] == "val")
    yp = os.path.join(d, "data.yaml")
    return {"dir": d, "yaml": yp if os.path.exists(yp) else "", "count": len(items),
            "train": len(items) - n_val, "val": n_val, "is_trainset": False,
            "training": _training_uses(d)}


def _trainset_sync(img_path):
    """這張圖已經在訓練資料夾裡 → 一般「儲存標註」或「復原」之後，把最新的標籤同步過去。
    復原到「沒有標籤檔」的狀態，就代表這張還沒修好，從訓練資料夾拿掉。"""
    if not SESSION["open"] or _is_trainset_session():
        return
    d = trainset_dir()
    m = _trainset_manifest(d)
    it = m["items"].get(os.path.abspath(img_path))
    if not it:
        return
    lp = label_path_for(img_path)
    if not os.path.isfile(lp):
        _trainset_drop(img_path)
        return
    _atomic_copy(lp, _trainset_paths(d, it)[1])
    it["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_trainset_manifest(d, m)


def _trainset_drop(img_path):
    """把一張原圖的複本從訓練資料夾拿掉（原圖被移除、移到測試資料夾時呼叫）。"""
    if not SESSION["open"] or _is_trainset_session():
        return False
    d = trainset_dir()
    m = _trainset_manifest(d)
    it = m["items"].pop(os.path.abspath(img_path), None)
    if not it:
        return False
    for f in _trainset_paths(d, it):
        if os.path.exists(f):
            os.remove(f)
    _write_trainset_manifest(d, m)
    _write_trainset_yaml(d)
    log(f"trainset drop: {img_path}")
    return True


def _training_uses(root):
    """訓練進行中，而且開始訓練時開著的就是這個資料夾的資料集 → 它的檔案正在被讀。"""
    return TRAIN["state"] == "running" and TRAIN.get("root") == os.path.abspath(root)


def _guard_image_removal(paths):
    """搬走／移除圖片前的檢查：正在被訓練讀取的檔案不能動，否則訓練會讀不到檔案而中斷。
    訓練的是別的資料夾（例如訓練資料夾）就不擋——可以一邊訓練一邊繼續修正原本的資料集。"""
    if _training_uses(SESSION["root"]):
        raise HTTPException(409, "訓練進行中，搬走圖片會讓訓練讀不到檔案而中斷，請等訓練結束再移動或移除")
    d = trainset_dir()
    if _training_uses(d):
        items = _trainset_manifest(d)["items"]
        if any(os.path.abspath(p) in items for p in paths):
            raise HTTPException(409, "有圖片也在訓練資料夾裡，而訓練資料夾正在訓練中；請等訓練結束再移動或移除這些圖片")


@app.get("/dataset/trainset")
def dataset_trainset():
    return _trainset_info()


@app.post("/dataset/trainset/add")
def dataset_trainset_add(payload: dict):
    """把一張（已存檔的）圖片與標籤「複製」到訓練資料夾，原始資料集不動。已經在裡面就更新標籤。
    background=true：這張沒有任何框，當成沒有目標的背景圖——磁碟上還沒有標籤檔就先存一個空的（會備份）。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    if _is_trainset_session():
        raise HTTPException(400, "目前開著的就是訓練資料夾本身，請先回到原本的資料集再加入")
    raw = (payload.get("path") or "").strip()
    match = next((p for p in SESSION["images"]
                  if p == raw or os.path.abspath(p) == os.path.abspath(raw)), None)
    if not match:
        raise HTTPException(404, "這張圖片不在目前的資料集")
    lp = label_path_for(match)
    if not os.path.isfile(lp):
        if not payload.get("background"):
            raise HTTPException(400, "這張圖還沒有標籤檔，請先儲存標註再加入訓練資料夾")
        make_backup(match)
        write_labels(match, [], SESSION["classes"])

    d = trainset_dir()
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    m = _trainset_manifest(d)
    key = os.path.abspath(match)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    it = m["items"].get(key)
    added = it is None
    if added:
        split = _pick_trainset_split(m, os.path.basename(match))
        name = _free_name(os.path.join(d, "images", split), os.path.join(d, "labels", split),
                          os.path.basename(match))
        it = m["items"][key] = {"split": split, "name": name, "added_at": now}
    img_dst, lbl_dst = _trainset_paths(d, it)
    if added or not os.path.exists(img_dst):
        _atomic_copy(match, img_dst)
    _atomic_copy(lp, lbl_dst)
    it["updated_at"] = now
    m["source_root"] = SESSION["root"]
    _write_trainset_manifest(d, m)
    _write_trainset_yaml(d)
    log(f"trainset {'add' if added else 'update'}: {match} -> {img_dst}")
    return {"status": "ok", "added": added, "split": it["split"], "trainset": _trainset_info()}


# =========================
# 檔案 / 圖片
# =========================
# tkinter 必須在自己的主執行緒跑；放在 thread pool 裡第二次呼叫就會噴
# "main thread is not in main loop"。所以每次都開一個短命子行程來顯示對話框。
_DIALOG_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog

kind  = sys.argv[1]
start = sys.argv[2] if len(sys.argv) > 2 else ""
geom  = sys.argv[3] if len(sys.argv) > 3 else "1500x950"

root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)

# Tk 內建的檔案對話框預設很小（約 480x360），也沒有 API 可以指定大小。
# 它其實是 root 底下一個叫 .__tk_filedialog 的 toplevel，等它「建好而且已經顯示」
# 之後再改 geometry 就能撐大；中間的檔案清單會跟著變大。
#
# 兩個關鍵：
#   1. 一定要等 winfo ismapped，太早改會在對話框還在組裝時丟出 Tcl 錯誤，
#      那個錯誤會打斷對話框自己的事件迴圈，讓它立刻以「取消」收場。
#   2. 全部包在 Tcl 的 catch 裡，任何錯誤都不會外流到對話框的事件迴圈。
def enlarge(tries=0):
    for name in (".__tk_filedialog", ".__tk_choosedir"):
        try:
            ready = root.tk.eval(
                "expr {[winfo exists %s] && [winfo ismapped %s]}" % (name, name))
        except tk.TclError:
            ready = "0"
        if ready == "1":
            root.tk.eval("catch {wm geometry %s %s}" % (name, geom))
            root.tk.eval("catch {wm minsize %s 800 500}" % name)
            return
    if tries < 400:
        root.after(30, enlarge, tries + 1)

root.after(30, enlarge)

kw = {}
if start:
    kw["initialdir"] = start

if kind == "folder":
    p = filedialog.askdirectory(title="選擇圖片資料夾", **kw)
elif kind == "image":
    p = filedialog.askopenfilename(title="選擇圖片",
        filetypes=[("圖片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有檔案", "*.*")], **kw)
else:
    p = filedialog.askopenfilename(title="選擇 data.yaml",
        filetypes=[("YOLO 資料集設定", "*.yaml *.yml"), ("所有檔案", "*.*")], **kw)
root.destroy()
sys.stdout.write(p or "")
"""


def _dialog(kind, start=""):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return {"status": "unsupported", "path": "",
                "message": "此環境沒有桌面（DISPLAY 未設定），請直接把路徑貼進輸入框"}
    # start 可以是檔案或資料夾；是檔案就取它所在的資料夾當起點
    if start:
        start = os.path.expanduser(start)
        if os.path.isfile(start):
            start = os.path.dirname(start)
        if not os.path.isdir(start):
            start = ""
    geom = str(CONFIG.get("native_dialog_size", "1500x950"))
    try:
        r = subprocess.run([sys.executable, "-c", _DIALOG_SCRIPT, kind, start, geom],
                           capture_output=True, text=True, timeout=300)
    except Exception as e:
        log(f"無法開啟選取視窗：{e}", "WARNING")
        return {"status": "unsupported", "path": "", "message": f"無法開啟選取視窗：{e}，請直接輸入路徑"}
    p = (r.stdout or "").strip()
    if p:
        p = os.path.normpath(p)
        log(f"dialog({kind}) 選擇 {p}")
        return {"status": "ok", "path": p}
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()[-1:] or ["未知錯誤"]
        log(f"選取視窗失敗：{err[0]}", "WARNING")
        return {"status": "unsupported", "path": "", "message": f"選取視窗無法開啟：{err[0]}"}
    return {"status": "cancelled", "path": "", "message": "已取消選取"}


FS_KINDS = {"yaml": (".yaml", ".yml"), "image": IMAGE_EXTS, "any": None}


@app.get("/fs/list")
def fs_list(path: str = "", kind: str = "yaml"):
    """列出資料夾內容，給前端自己的檔案瀏覽器用。
    比原生對話框好的地方：視窗大小自己決定、遠端從別台機器開也能用、樣式一致。"""
    root = os.path.abspath(os.path.expanduser(path)) if path else os.path.expanduser("~")
    if not os.path.isdir(root):
        root = os.path.dirname(root) if os.path.isfile(root) else os.path.expanduser("~")
    exts = FS_KINDS.get(kind, FS_KINDS["yaml"])

    dirs, files = [], []
    try:
        for name in sorted(os.listdir(root), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            try:
                if os.path.isdir(full):
                    dirs.append({"name": name, "path": full})
                elif exts is None or name.lower().endswith(tuple(exts)):
                    files.append({"name": name, "path": full, "size": os.path.getsize(full)})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"沒有權限讀取：{root}")
    except OSError as e:
        raise HTTPException(400, f"無法讀取資料夾：{e}")

    home = os.path.expanduser("~")
    shortcuts = [{"name": "家目錄", "path": home}]
    for label, sub in (("桌面", "Desktop"), ("下載", "Downloads")):
        d = os.path.join(home, sub)
        if os.path.isdir(d):
            shortcuts.append({"name": label, "path": d})
    shortcuts.append({"name": "本程式目錄", "path": PKG_DIR})
    if SESSION.get("open") and SESSION.get("root"):
        shortcuts.append({"name": "目前資料集", "path": SESSION["root"]})

    parent = os.path.dirname(root.rstrip(os.sep)) or "/"
    crumbs, acc = [], ""
    for part in root.strip(os.sep).split(os.sep):
        acc += os.sep + part
        crumbs.append({"name": part, "path": acc})
    return {"path": root, "parent": parent if parent != root else "",
            "crumbs": crumbs, "dirs": dirs, "files": files, "shortcuts": shortcuts}


@app.get("/pick-folder")
def pick_folder(start: str = ""):
    return _dialog("folder", start)


@app.get("/pick-file")
def pick_file(kind: str = "yaml", start: str = ""):
    """kind=yaml 選 data.yaml；kind=image 選單張圖片。start 是對話框的起始資料夾。"""
    return _dialog("image" if kind == "image" else "yaml", start)


@app.get("/image")
def image(path: str, w: int = 0):
    """w>0 回傳快取縮圖。標註頁的縮圖牆走這條，瀏覽器不必解 4K 原圖。"""
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    if w <= 0:
        return FileResponse(path, headers={"Cache-Control": "max-age=3600"})
    w = max(48, min(w, 640))
    key = hashlib.md5(f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{w}".encode()).hexdigest()
    thumb = os.path.join(THUMB_DIR, key + ".jpg")
    if not os.path.exists(thumb):
        try:
            with Image.open(path) as im:
                im.draft("RGB", (w * 2, w * 2))       # JPEG 解碼時就降尺寸，省記憶體與時間
                im = im.convert("RGB")
                im.thumbnail((w, w), Image.BILINEAR)
                im.save(thumb, "JPEG", quality=72)
        except Exception:
            return FileResponse(path, headers={"Cache-Control": "max-age=3600"})
    return FileResponse(thumb, headers={"Cache-Control": "max-age=86400"})


@app.get("/weights")
def weights():
    try:
        return {"weights": sorted(f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt"))}
    except OSError:
        return {"weights": []}


@app.get("/models")
def models():
    """依修改時間新→舊排序：剛訓練好的模型排第一個，推論頁預設就會選到它。
    engines 是 TensorRT 產出的 .engine（含建立時的 imgsz / 精度資訊）。"""
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True)
        engines = []
        for f in sorted((f for f in os.listdir(MODELS_DIR) if f.endswith(".engine")),
                        key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)), reverse=True):
            meta = {}
            mp = os.path.join(MODELS_DIR, f + ".json")
            if os.path.exists(mp):
                try:
                    with open(mp, encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = {}
            engines.append({"name": f, "imgsz": meta.get("imgsz"), "half": meta.get("half", True),
                            "size_mb": meta.get("size_mb"), "tensorrt": meta.get("tensorrt")})
        return {"models": files, "engines": engines}
    except OSError:
        return {"models": [], "engines": []}


# =========================
# 訓練（子行程）
# =========================
TRAIN = {"state": "idle", "proc": None, "logs": deque(maxlen=400), "epoch": 0, "epochs": 0,
         "metrics": {}, "name": "", "save_dir": "", "started": 0, "error": "",
         "phase": "", "job": None, "engine": ""}
TRAIN_LOCK = threading.Lock()


def stratified_split(images, classes, val_ratio=0.2):
    """依每張圖的主要類別分層 8:2；只有 1 張的類別全放 train。回傳 (train, val, stats)。"""
    import random
    groups = {}
    for p in images:
        boxes = read_labels(p, classes)
        if not boxes:
            key = "__bg__"
        else:
            counts = {}
            for b in boxes:
                counts[b["className"]] = counts.get(b["className"], 0) + 1
            key = max(counts, key=counts.get)
        groups.setdefault(key, []).append(p)
    train, val, stats = [], [], {}
    rnd = random.Random(0)
    for key, items in groups.items():
        rnd.shuffle(items)
        if len(items) < 2:
            train += items
            stats[key] = {"total": len(items), "train": len(items), "val": 0}
        else:
            cut = max(1, int(len(items) * (1 - val_ratio)))
            train += items[:cut]
            val += items[cut:]
            stats[key] = {"total": len(items), "train": cut, "val": len(items) - cut}
    return train, val, stats


def stratified_split3(images, classes, val_ratio, test_ratio):
    """依每張圖的主要類別分層切成 train/val/test。樣本太少的類別會先犧牲 test、
    再犧牲 val，至少留 1 張在 train。回傳 (train, val, test, stats)。"""
    import random
    groups = {}
    for p in images:
        boxes = read_labels(p, classes)
        if not boxes:
            key = "__bg__"
        else:
            counts = {}
            for b in boxes:
                counts[b["className"]] = counts.get(b["className"], 0) + 1
            key = max(counts, key=counts.get)
        groups.setdefault(key, []).append(p)

    train, val, test, stats = [], [], [], {}
    rnd = random.Random(0)
    for key, items in groups.items():
        rnd.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0
        while n_test + n_val >= n and n_test > 0:
            n_test -= 1
        while n_test + n_val >= n and n_val > 0:
            n_val -= 1
        n_train = n - n_val - n_test
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
        stats[key] = {"total": n, "train": n_train, "val": n_val, "test": n_test}
    return train, val, test, stats


def build_train_data(name):
    """回傳 (data_path, dataset_stats)。yaml/merged/split 模式直接用各自的 data.yaml；
    資料夾模式產生 train.txt / val.txt 絕對路徑清單 + data.yaml —— 完全不複製任何一張圖。"""
    if SESSION["mode"] in ("yaml", "merged", "split") and SESSION["yaml"]:
        labeled = sum(1 for p in SESSION["images"] if os.path.exists(label_path_for(p)))
        src = {"merged": "merged data.yaml", "split": "train/val/test 切分"}.get(SESSION["mode"], "data.yaml")
        return SESSION["yaml"], {"total": len(SESSION["images"]), "labeled": labeled, "source": src}

    job_dir = os.path.join(JOB_DIR, name)
    os.makedirs(job_dir, exist_ok=True)
    train, val, stats = stratified_split(SESSION["images"], SESSION["classes"])
    for fname, items in (("train.txt", train), ("val.txt", val or train[-1:])):
        with open(os.path.join(job_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")
    data_path = os.path.join(job_dir, "data.yaml")
    with open(data_path, "w", encoding="utf-8") as f:
        yaml.dump({"path": job_dir, "train": os.path.join(job_dir, "train.txt"),
                   "val": os.path.join(job_dir, "val.txt"),
                   "names": {i: n for i, n in enumerate(SESSION["classes"])},
                   "nc": len(SESSION["classes"])}, f, allow_unicode=True, sort_keys=False)
    return data_path, {"total": len(SESSION["images"]), "train": len(train), "val": len(val),
                       "classes": stats, "source": "folder"}


# =========================
# 資料集頁：把已標註影像切成 train/val/test（產生獨立 data.yaml，原始資料不動）
# =========================
SPLITS_DIR = os.path.join(OUTPUT_DIR, ".splits")
os.makedirs(SPLITS_DIR, exist_ok=True)


@app.post("/dataset/split")
async def dataset_split_apply(payload: dict):
    """對目前「已標註」影像做分層切分，產生獨立的 data.yaml + train/val/test.txt。
    套用後訓練自動改用這份切分；使用者原本的 data.yaml／資料夾完全不會被改動。"""
    if not SESSION["open"]:
        raise HTTPException(400, "請先載入目前的資料集")
    if not SESSION["classes"]:
        raise HTTPException(400, "資料集沒有類別，無法切分")
    if TRAIN["state"] == "running":
        raise HTTPException(409, "訓練進行中，請結束後再切分")
    try:
        val_pct = float(payload.get("val_pct", 10))
        test_pct = float(payload.get("test_pct", 5))
    except (TypeError, ValueError):
        raise HTTPException(400, "百分比格式錯誤")
    if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
        raise HTTPException(400, "Val + Test 的百分比總和必須小於 100")

    labeled = [p for p in SESSION["images"] if os.path.exists(label_path_for(p))]
    if len(labeled) < 10:
        raise HTTPException(400, f"已標註影像太少（{len(labeled)} 張），建議至少 10 張再切分")

    train, val, test, stats = stratified_split3(labeled, SESSION["classes"], val_pct / 100, test_pct / 100)

    split_dir = os.path.join(SPLITS_DIR, "active")
    shutil.rmtree(split_dir, ignore_errors=True)
    os.makedirs(split_dir, exist_ok=True)
    for fname, items in (("train.txt", train), ("val.txt", val or train[-1:]), ("test.txt", test)):
        with open(os.path.join(split_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + ("\n" if items else ""))
    info = {"path": split_dir, "train": os.path.join(split_dir, "train.txt"),
            "val": os.path.join(split_dir, "val.txt"),
            "names": {i: n for i, n in enumerate(SESSION["classes"])}, "nc": len(SESSION["classes"])}
    if test:
        info["test"] = os.path.join(split_dir, "test.txt")
    data_yaml = os.path.join(split_dir, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(info, f, allow_unicode=True, sort_keys=False)

    if not SESSION.get("split"):        # 只在「第一次套用」時記住原本的來源，重複套用不能覆蓋掉它
        SESSION["split_pre_yaml"] = SESSION["yaml"]
        SESSION["split_pre_mode"] = SESSION["mode"]
    SESSION["yaml"], SESSION["mode"] = data_yaml, "split"
    SESSION["split"] = {"val_pct": val_pct, "test_pct": test_pct, "yaml": data_yaml,
                         "counts": {"train": len(train), "val": len(val), "test": len(test)},
                         "labeled_total": len(labeled), "class_stats": stats,
                         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    log(f"split applied: train={len(train)} val={len(val)} test={len(test)} "
        f"val_pct={val_pct} test_pct={test_pct}")
    return {"status": "ok", "split": SESSION["split"], "summary": dataset_summary()}


@app.post("/dataset/split/revert")
async def dataset_split_revert():
    """撤銷切分，訓練資料來源改回套用切分之前的狀態（原始 yaml、資料夾、或合併後的資料集）。"""
    if not SESSION.get("split"):
        raise HTTPException(400, "目前沒有套用中的切分")
    SESSION["yaml"] = SESSION.get("split_pre_yaml") or ""
    SESSION["mode"] = SESSION.get("split_pre_mode") or "folder"
    SESSION["split"] = None
    shutil.rmtree(os.path.join(SPLITS_DIR, "active"), ignore_errors=True)
    log("split reverted")
    return {"status": "ok", "summary": dataset_summary()}


@app.post("/dataset/split/evaluate")
def dataset_split_evaluate(payload: dict):
    """在切分出來的 held-out test 集上跑一次 model.val()（不是訓練）。同步子行程呼叫，
    這個 route 是同步 def，FastAPI 會丟到執行緒池跑，不會卡住其他請求；通常幾十秒內完成。"""
    split = SESSION.get("split")
    if not split or not split["counts"].get("test"):
        raise HTTPException(400, "目前沒有套用切分，或切分沒有 test 集")
    if TRAIN["state"] == "running":
        raise HTTPException(409, "訓練進行中，請結束後再評估")
    model = (payload.get("model") or "").strip()
    metrics = _run_test_eval(model, split["yaml"], int(payload.get("imgsz", 640)))
    return {"status": "ok", "model": model, "metrics": metrics, "images": split["counts"].get("test", 0)}


@app.post("/dataset/testset/evaluate")
def dataset_testset_evaluate(payload: dict):
    """在測試圖片資料夾（標註頁手動移出去的圖）上跑 model.val()。只計已標註的圖。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    if TRAIN["state"] == "running":
        raise HTTPException(409, "訓練進行中，請結束後再評估")
    data_yaml, n = _write_testset_yaml(testset_dir())   # 每次重寫：類別可能改過名
    if not n:
        raise HTTPException(400, "測試圖片資料夾裡沒有已標註的圖片，無法計算 mAP")
    model = (payload.get("model") or "").strip()
    metrics = _run_test_eval(model, data_yaml, int(payload.get("imgsz", 640)))
    return {"status": "ok", "model": model, "metrics": metrics, "images": n}


def _run_test_eval(model, data_yaml, imgsz):
    """跑 eval_worker（model.val(split="test")）並回傳 metrics；失敗丟 HTTPException。"""
    mp = os.path.join(MODELS_DIR, model)
    if not os.path.exists(mp):
        mp = os.path.join(WEIGHTS_DIR, model)
    if not model or not os.path.exists(mp):
        raise HTTPException(400, f"找不到模型：{model}")
    job = {"model": mp, "data": data_yaml, "imgsz": imgsz}
    jf = os.path.join(SPLITS_DIR, "eval_job.json")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    # YOLO_AUTOINSTALL=false：跟訓練一樣，不讓 ultralytics 在建資料集時自己 pip 裝套件（例如 pi-heif 會順手升級 Pillow）
    env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
    try:
        r = subprocess.run([sys.executable, "-u", os.path.join(WORKER_DIR, "eval_worker.py"), jf],
                           capture_output=True, text=True, timeout=600, env=env, cwd=WEIGHTS_DIR)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "評估逾時（10 分鐘），模型或測試集可能太大")
    for line in (r.stdout or "").splitlines():
        if not line.startswith("@@"):
            continue
        try:
            ev = json.loads(line[2:])
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "done":
            log(f"test-set eval: model={model} data={data_yaml} metrics={ev.get('metrics')}")
            return ev.get("metrics", {})
        if ev.get("event") == "error":
            raise HTTPException(500, f"評估失敗：{ev.get('message')}")
    raise HTTPException(500, "評估行程沒有回傳結果：" + (r.stderr or "")[-800:])


def tail_process(proc):
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip("\n")
        if line.startswith("@@"):
            try:
                ev = json.loads(line[2:])
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "epoch":
                TRAIN["epoch"], TRAIN["epochs"] = ev.get("epoch", 0), ev.get("epochs", 0)
                TRAIN["metrics"] = ev.get("metrics", {})
                TRAIN["logs"].append(f"[EPOCH] {TRAIN['epoch']}/{TRAIN['epochs']} "
                                     + "  ".join(f"{k}={v}" for k, v in TRAIN["metrics"].items()))
            elif kind == "done":
                TRAIN["save_dir"] = ev.get("save_dir", "")
                TRAIN["metrics"] = ev.get("metrics", {})
                TRAIN["logs"].append(f"[DONE] 模型已存至 {ev.get('model_path') or ev.get('save_dir')}")
            elif kind == "error":
                TRAIN["error"] = ev.get("message", "")
                TRAIN["logs"].append(f"[ERROR] {TRAIN['error']}")
            elif kind == "device":
                TRAIN["logs"].append(f"[SYSTEM] device={ev.get('device')} cuda={ev.get('cuda')}")
            elif kind == "start":
                TRAIN["logs"].append(f"[SYSTEM] 載入 {os.path.basename(ev.get('model',''))}，"
                                     f"資料集 {ev.get('data')}，{ev.get('epochs')} epochs")
        elif line.strip():
            TRAIN["logs"].append(line)
    code = proc.wait()
    ok = code == 0
    TRAIN["state"] = "done" if ok else ("stopped" if code in (-15, -9, 143, 137) else "error")
    if not ok and not TRAIN["error"]:
        TRAIN["error"] = f"訓練行程結束碼 {code}"
    TRAIN["logs"].append(f"[SYSTEM] 訓練行程結束（code={code}），記憶體已釋放")
    log(f"train finished: state={TRAIN['state']} code={code}")

    job = TRAIN.get("job") or {}
    if ok and job.get("tensorrt"):
        start_export(job)


def start_export(job):
    """訓練完成後轉 TensorRT。等訓練行程結束才開始，記憶體才夠 TensorRT 建圖。"""
    src = os.path.join(MODELS_DIR, job["name"] + ".pt")
    if not os.path.exists(src):
        TRAIN["logs"].append(f"[ERROR] 找不到剛訓練好的模型：{src}，略過 TensorRT 匯出")
        return
    ejob = {"model": src, "name": job["name"], "models_dir": MODELS_DIR,
            "imgsz": job.get("imgsz", 640), "half": True}
    jf = os.path.join(JOB_DIR, f"{job['name']}_export.json")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(ejob, f, ensure_ascii=False)

    TRAIN["state"] = "running"
    TRAIN["phase"] = "export"
    TRAIN["logs"].append("[SYSTEM] 開始轉換 TensorRT（FP16）——這一步比訓練慢，過程中畫面看起來像卡住是正常的")
    # YOLO_AUTOINSTALL=false：離線的 Jetson 上 ultralytics 會嘗試 pip 裝 onnxruntime-gpu，
    # 一定失敗又浪費時間，直接關掉
    env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(WORKER_DIR, "export_worker.py"), jf],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env, cwd=WEIGHTS_DIR, start_new_session=True)
    TRAIN["proc"] = proc
    threading.Thread(target=tail_export, args=(proc,), daemon=True).start()
    log(f"tensorrt export started: {src} (PID {proc.pid})")


def tail_export(proc):
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip("\n")
        if line.startswith("@@"):
            try:
                ev = json.loads(line[2:])
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "done":
                TRAIN["engine"] = ev.get("path", "")
                TRAIN["logs"].append(
                    f"[DONE] TensorRT engine 已產生：{ev.get('path')}"
                    f"（{ev.get('size_mb')} MB, imgsz={ev.get('imgsz')}, "
                    f"{'FP16' if ev.get('half') else 'FP32'}, TensorRT {ev.get('tensorrt')}）")
            elif kind == "error":
                TRAIN["error"] = ev.get("message", "")
                TRAIN["logs"].append(f"[ERROR] TensorRT 匯出失敗：{TRAIN['error']}")
            elif kind in ("info", "start"):
                TRAIN["logs"].append(f"[SYSTEM] {ev.get('message') or ev}")
        elif line.strip():
            TRAIN["logs"].append(line)
    code = proc.wait()
    TRAIN["phase"] = ""
    # 訓練本身已經成功，匯出失敗不該讓整個結果看起來像失敗
    TRAIN["state"] = "done" if code == 0 else "export_failed"
    TRAIN["logs"].append(f"[SYSTEM] TensorRT 匯出行程結束（code={code}）")
    log(f"tensorrt export finished: code={code}")


@app.post("/train")
async def train(payload: dict):
    with TRAIN_LOCK:
        if TRAIN["state"] == "running":
            raise HTTPException(409, "已有訓練進行中")
        if not SESSION["open"]:
            raise HTTPException(400, "請先開啟資料集")
        if not SESSION["classes"]:
            raise HTTPException(400, "資料集沒有類別（names），無法訓練")

        model = payload.get("model") or "yolo11n.pt"
        model_path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(model_path):
            model_path = os.path.join(WEIGHTS_DIR, model)
        if not os.path.exists(model_path):
            raise HTTPException(400, f"找不到權重：{model}")

        name = (payload.get("name") or "").strip() or f"lite_{datetime.now():%m%d_%H%M}"
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        data_path, stats = build_train_data(name)

        job = {"tensorrt": bool(payload.get("tensorrt")),
               "model": model_path, "data": data_path, "epochs": int(payload.get("epochs", 50)),
               "imgsz": int(payload.get("imgsz", 640)), "batch": int(payload.get("batch", 8)),
               "workers": int(payload.get("workers", 2)), "device": payload.get("device") or "",
               "project": RUNS_DIR, "name": name, "models_dir": MODELS_DIR,
               "dataset_stats": stats, "aug": payload.get("aug") or {}}
        job_file = os.path.join(JOB_DIR, f"{name}.json")
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)

        # YOLO_AUTOINSTALL=false：ultralytics 建資料集時會想 pip 裝 pi-heif 之類的套件（還會順手升級 Pillow）
        env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
        # cwd 設在 weights/：Ultralytics 的 AMP 檢查會去 cwd 找 yolo11n.pt，
        # 指到這裡就直接用包內既有權重，離線環境也不會嘗試連網下載。
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(WORKER_DIR, "train_worker.py"), job_file],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env, cwd=WEIGHTS_DIR, start_new_session=True)
        TRAIN.update({"state": "running", "proc": proc, "epoch": 0, "epochs": job["epochs"],
                      "metrics": {}, "name": name, "save_dir": "", "started": time.time(), "error": "",
                      "phase": "train", "job": job, "engine": "",
                      "root": os.path.abspath(SESSION["root"])})   # 這個資料集的檔案在訓練期間不能搬走
        TRAIN["logs"].clear()
        TRAIN["logs"].append(f"[SYSTEM] 訓練子行程已啟動（PID {proc.pid}），主 server 不載入 torch")
        threading.Thread(target=tail_process, args=(proc,), daemon=True).start()
        log(f"train started: name={name} model={model} data={data_path}")
        return {"status": "ok", "name": name, "data": data_path, "dataset": stats}


@app.get("/train/status")
def train_status(offset: int = 0):
    logs = list(TRAIN["logs"])
    return {"state": TRAIN["state"], "phase": TRAIN["phase"], "engine": TRAIN["engine"],
            "tensorrt": bool((TRAIN.get("job") or {}).get("tensorrt")),
            "name": TRAIN["name"], "epoch": TRAIN["epoch"],
            "epochs": TRAIN["epochs"], "metrics": TRAIN["metrics"], "error": TRAIN["error"],
            "save_dir": TRAIN["save_dir"], "elapsed": int(time.time() - TRAIN["started"]) if TRAIN["started"] else 0,
            "logs": logs[offset:], "log_total": len(logs)}


@app.post("/train/stop")
def train_stop():
    proc = TRAIN.get("proc")
    if proc and proc.poll() is None:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        TRAIN["logs"].append("[SYSTEM] 已送出停止訊號")
        return {"status": "ok"}
    return {"status": "idle"}


# =========================
# 評估：直接讀訓練產出的 metrics.json 與 Ultralytics 自己畫的圖
# =========================
@app.get("/runs")
def runs():
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        d = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(d) or name.startswith((".", "datasets", "logs", "models", "dataset_temp")):
            continue
        if not (os.path.exists(os.path.join(d, "results.csv")) or os.path.exists(os.path.join(d, "metrics.json"))):
            continue
        out.append({"name": name, "mtime": int(os.path.getmtime(d))})
    return {"runs": out}


@app.get("/run")
def run_detail(name: str):
    d = os.path.join(RUNS_DIR, os.path.basename(name))
    if not os.path.isdir(d):
        raise HTTPException(404, "找不到訓練結果")
    metrics = {}
    mp = os.path.join(d, "metrics.json")
    if os.path.exists(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass
    plots = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg"))]
    return {"name": os.path.basename(name), "dir": d, "metrics": metrics, "plots": plots,
            "args": os.path.exists(os.path.join(d, "args.yaml"))}


# =========================
# 推論（常駐但會自己過期的子行程）
# =========================
INFER = {"proc": None}
INFER_LOCK = threading.Lock()


def infer_proc():
    p = INFER.get("proc")
    if p and p.poll() is None:
        return p
    env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false",
               LITE_INFER_IDLE=str(CONFIG.get("infer_idle_seconds", 180)))
    p = subprocess.Popen([sys.executable, "-u", os.path.join(WORKER_DIR, "infer_worker.py")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1, env=env, cwd=WEIGHTS_DIR)
    p.stdout.readline()          # ready
    INFER["proc"] = p
    log(f"infer worker started (PID {p.pid}), idle timeout {env['LITE_INFER_IDLE']}s")
    return p


@app.post("/predict")
def predict(payload: dict):
    """同步 def（不是 async）：FastAPI 會丟到 thread pool 跑。第一次呼叫要載入 torch + 模型，
    要好幾秒，若寫成 async def 會把整個 event loop 卡住（縮圖、系統狀態全部停擺）。
    INFER_LOCK 是 threading.Lock，多個請求照樣一個一個排隊進 worker。"""
    path = payload.get("path") or ""
    model = payload.get("model") or ""
    if os.path.isdir(path):
        raise HTTPException(400, "這是資料夾，請選到單一張圖片檔")
    if not os.path.isfile(path):
        raise HTTPException(404, f"找不到圖片：{path}")
    mp = os.path.join(MODELS_DIR, model)
    if not os.path.exists(mp):
        mp = os.path.join(WEIGHTS_DIR, model)
    if not os.path.exists(mp):
        raise HTTPException(400, f"找不到模型：{model}")
    req = json.dumps({"cmd": "predict", "model": mp, "path": path,
                      "conf": float(payload.get("conf", 0.25)),
                      "half": bool(payload.get("half", False)),
                      "imgsz": int(payload.get("imgsz", 640))}) + "\n"
    with INFER_LOCK:
        for attempt in (1, 2):                     # worker 可能剛好閒置逾時結束 → 重試一次
            try:
                p = infer_proc()
                p.stdin.write(req)
                p.stdin.flush()
                # 正常情況下 worker 只會吐 JSON，但萬一有函式庫繞過去印了東西，
                # 就跳過那幾行繼續找，不要整個請求失敗
                for _ in range(20):
                    line = p.stdout.readline()
                    if not line:
                        raise BrokenPipeError("worker no response")
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        log(f"infer worker 非預期輸出：{line[:120]}", "WARNING")
                raise BrokenPipeError("worker 只回了無法解析的內容")
            except (BrokenPipeError, OSError, json.JSONDecodeError) as e:
                INFER["proc"] = None
                if attempt == 2:
                    raise HTTPException(500, f"推論失敗：{e}")


@app.post("/predict/release")
def predict_release():
    """手動釋放推論模型記憶體。"""
    p = INFER.get("proc")
    if p and p.poll() is None:
        try:
            p.stdin.write(json.dumps({"cmd": "bye"}) + "\n")
            p.stdin.flush()
            p.wait(timeout=10)
        except Exception:
            p.kill()
    INFER["proc"] = None
    return {"status": "ok"}


# =========================
# 推論頁：批次推論整個資料夾，效果不好的圖一鍵加入目前資料集的訓練資料
# =========================
# 紀錄哪些圖是從推論頁加入的（依資料集 root 分開），標註頁靠它篩出「從推論加入」的圖。
# 寫成檔案而不是只放 SESSION，server 重開後篩選照樣有效。
INFER_ADDED_FILE = os.path.join(OUTPUT_DIR, ".infer_added.json")


def _infer_added_all():
    try:
        with open(INFER_ADDED_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _infer_added_save(d):
    with open(INFER_ADDED_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def _infer_added_mine():
    return _infer_added_all().get(SESSION.get("root") or "", []) if SESSION["open"] else []


def _infer_added_mark_reviewed(path):
    """標註頁存檔時呼叫：從推論加入的圖一旦存過一次，就視為使用者已經確認過預標註。"""
    if not SESSION["open"]:
        return
    store = _infer_added_all()
    hit = False
    for e in store.get(SESSION["root"], []):
        if e["path"] == path and not e.get("reviewed"):
            e["reviewed"] = True
            hit = True
    if hit:
        _infer_added_save(store)


def _infer_added_pending():
    """複製進來、檔案還在、但還沒在標註頁存檔確認過的圖片數。"""
    in_ds = set(SESSION["images"])
    return sum(1 for e in _infer_added_mine()
               if e.get("copied") and not e.get("reviewed") and e["path"] in in_ds)


def _add_target_dir():
    """加入的圖要複製到哪裡，訓練才會用到：yaml/合併模式 → 原始 data.yaml 的第一個 train 資料夾；
    資料夾模式 → 該資料夾本身；切分模式 → 依切分前的來源判斷（另外會把圖補進切分的 train.txt）。"""
    mode, ypath = SESSION["mode"], SESSION["yaml"]
    if mode == "split":
        mode, ypath = SESSION.get("split_pre_mode") or "folder", SESSION.get("split_pre_yaml") or ""
    if mode == "merged":
        ypath = SESSION.get("orig_yaml") or ypath
    if mode in ("yaml", "merged") and ypath:
        try:
            with open(ypath, encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
        except Exception as e:
            raise HTTPException(400, f"讀取 data.yaml 失敗：{e}")
        dirs, _ = dataset_dirs_from_yaml(ypath, info, "train")
        if not dirs:
            raise HTTPException(400, "data.yaml 的 train 不是資料夾（可能是圖片清單檔），無法判斷要加到哪裡")
        return dirs[0]
    return SESSION["root"]


def _norm_cls(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


@app.get("/infer/folder")
def infer_folder(path: str):
    """列出資料夾內的圖片給批次推論用（不遞迴）。資料夾本身沒有圖但底下有 images/ 時自動往下一層。"""
    path = path.strip()
    if not os.path.isdir(path):
        raise HTTPException(404, f"找不到資料夾：{path}")
    images = scan_images([path])
    if not images and os.path.isdir(os.path.join(path, "images")):
        path = os.path.join(path, "images")
        images = scan_images([path])
    if not images:
        raise HTTPException(400, "這個資料夾裡沒有圖片（支援 .jpg / .jpeg / .png / .bmp / .webp）")
    in_ds = {os.path.abspath(p) for p in SESSION["images"]} if SESSION["open"] else set()
    added = {e.get("src"): e["path"] for e in _infer_added_mine()}
    items = []
    for p in images:
        ap = os.path.abspath(p)
        it = {"path": p, "name": os.path.basename(p)}
        if ap in added and os.path.isfile(added[ap]):
            it["added"] = added[ap]                    # 之前已經從這張加入過
        elif ap in in_ds:
            it["in_dataset"] = True
        items.append(it)
    return {"folder": path, "items": items}


@app.get("/infer/add_target")
def infer_add_target():
    """告訴前端「加入訓練集」會把圖放到哪裡，按之前就看得到。"""
    if not SESSION["open"]:
        return {"open": False}
    try:
        d = _add_target_dir()
    except HTTPException as e:
        return {"open": True, "dir": "", "error": e.detail}
    # 順便猜 test 資料夾，批次推論的路徑欄位空著時直接帶入（使用者最常拿 test 集來測）
    test_dir = ""
    ypath = SESSION.get("orig_yaml") or SESSION.get("split_pre_yaml") or SESSION["yaml"]
    if ypath and os.path.isfile(ypath) and not ypath.startswith(SPLITS_DIR):
        try:
            with open(ypath, encoding="utf-8") as f:
                tdirs, _ = dataset_dirs_from_yaml(ypath, yaml.safe_load(f) or {}, "test")
            test_dir = tdirs[0] if tdirs else ""
        except Exception:
            pass
    return {"open": True, "dir": d, "mode": SESSION["mode"], "root": SESSION["root"], "test_dir": test_dir,
            "classes": SESSION["classes"], "count": len(_infer_added_mine()),
            "pending": _infer_added_pending()}


@app.post("/infer/add")
async def infer_add(payload: dict):
    """把推論測試的圖複製進目前資料集的訓練資料夾，模型預測框（依類別名稱對應）寫成預標註，
    之後到標註頁修正。原始圖片完全不動。沒有任何框能對應時不寫標籤檔，讓它顯示為「未標註」，
    免得被當成背景圖直接拿去訓練。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集，請先到「資料集」頁載入要加入的訓練資料集")
    src = os.path.abspath((payload.get("path") or "").strip())
    if not os.path.isfile(src):
        raise HTTPException(404, f"找不到圖片：{src}")
    store = _infer_added_all()
    mine = store.setdefault(SESSION["root"], [])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 這張本來就在資料集裡（例如直接拿訓練資料夾來批次測）→ 不複製，只記下來方便到標註頁找
    if any(os.path.abspath(p) == src for p in SESSION["images"]):
        if not any(e["path"] == src for e in mine):
            mine.append({"path": src, "src": src, "copied": False, "model": payload.get("model", ""),
                         "added_at": stamp})
            _infer_added_save(store)
        return {"status": "exists", "dest": src, "boxes": 0, "skipped_classes": [],
                "summary": dataset_summary()}
    prev = next((e for e in mine if e.get("src") == src and os.path.isfile(e["path"])), None)
    if prev:
        return {"status": "exists", "dest": prev["path"], "boxes": 0, "skipped_classes": [],
                "summary": dataset_summary()}

    dest_dir = _add_target_dir()
    os.makedirs(dest_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    dest, n = os.path.join(dest_dir, stem + ext), 1
    while os.path.exists(dest) or os.path.exists(label_path_for(dest)):   # 同名不覆蓋，改名
        dest = os.path.join(dest_dir, f"{stem}_inf{n}{ext}")
        n += 1
    shutil.copy2(src, dest)

    # 預測框 → 資料集類別（先比完全相同，再比忽略大小寫與 -/_/空白的寫法）
    exact = {c: c for c in SESSION["classes"]}
    loose = {_norm_cls(c): c for c in SESSION["classes"]}
    boxes, skipped = [], {}
    for d in payload.get("dets") or []:
        if d.get("type") != "bbox":
            continue
        cname = exact.get(d.get("cls")) or loose.get(_norm_cls(d.get("cls", "")))
        if not cname:
            skipped[d.get("cls")] = skipped.get(d.get("cls"), 0) + 1
            continue
        boxes.append({"className": cname, "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]})
    if boxes:
        write_labels(dest, boxes, SESSION["classes"])

    # 切分模式：訓練讀的是切分出來的 train.txt，新圖要補進去才會被用到
    if SESSION["mode"] == "split":
        _split_lists_edit(add={"train": [dest]})

    _reinsert_image(dest)
    mine.append({"path": dest, "src": src, "copied": True, "model": payload.get("model", ""),
                 "added_at": stamp})
    _infer_added_save(store)
    log(f"infer add: {src} -> {dest} boxes={len(boxes)} skipped={skipped}")
    return {"status": "ok", "dest": dest, "boxes": len(boxes), "skipped_classes": sorted(skipped),
            "summary": dataset_summary()}


@app.post("/infer/add/undo")
async def infer_add_undo(payload: dict):
    """撤銷加入：複製進來的那份圖＋標籤搬到 .removed_images/（跟標註頁的移除一樣，可再復原），
    本來就在資料集裡的圖只從「從推論加入」清單拿掉，檔案不動。"""
    if not SESSION["open"]:
        raise HTTPException(400, "尚未開啟資料集")
    dest = (payload.get("dest") or "").strip()
    store = _infer_added_all()
    mine = store.get(SESSION["root"], [])
    entry = next((e for e in mine if e["path"] == dest), None)
    if not entry:
        raise HTTPException(404, "這張圖不是從推論頁加入的，請到標註頁用「移除」")
    if entry.get("copied"):
        _guard_image_removal([dest])           # 這個資料集正在訓練中 → 不能把檔案搬走
    mine.remove(entry)
    if entry.get("copied"):
        rm = _remove_one_image(dest)
        if rm:
            stack = _removed_log()
            stack.append(rm)
            _write_removed_log(stack)
        SESSION["images"] = [p for p in SESSION["images"] if p != dest]
        if SESSION["mode"] == "split":
            _split_lists_edit(remove=[dest])
    _infer_added_save(store)
    log(f"infer add undo: {dest} copied={entry.get('copied')}")
    return {"status": "ok", "summary": dataset_summary()}


# =========================
# 系統
# =========================
@app.get("/")
def index():
    """回傳首頁，並依 app.js / style.css 的 mtime 掛上版本參數。
    改過前端就換一組 URL，瀏覽器不會拿舊的快取來用（不必叫使用者硬重整）。"""
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    for asset in ("style.css", "app.js"):
        fp = os.path.join(WEB_DIR, asset)
        ver = int(os.path.getmtime(fp)) if os.path.exists(fp) else 0
        html = html.replace(f"/web/{asset}", f"/web/{asset}?v={ver}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok", "licensed": LICENSED}


@app.get("/defaults")
def defaults():
    """前端啟動時拿預設值（例如常用的 data.yaml 路徑）。"""
    return {"default_yaml": CONFIG.get("default_yaml", ""),
            "default_folder": CONFIG.get("default_folder", ""),
            # native = 系統原生對話框（tkinter）；builtin = 網頁內建瀏覽器
            "file_picker": CONFIG.get("file_picker", "native"),
            "native_dialog_size": CONFIG.get("native_dialog_size", "1500x950")}


@app.get("/license/status")
def license_status():
    return {"licensed": LICENSED, "message": LICENSE_MSG, "mac": MACHINE_MAC}


@app.get("/sysinfo")
def sysinfo():
    """系統與本程式記憶體用量——輕量版的重點就是這個數字，所以直接顯示在畫面上。"""
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0]) // 1024      # MB
    except OSError:
        pass
    rss = 0
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except (OSError, IndexError):
        pass
    workers = []
    if TRAIN.get("proc") and TRAIN["proc"].poll() is None:
        workers.append({"kind": "train", "pid": TRAIN["proc"].pid, "rss": proc_rss(TRAIN["proc"].pid)})
    if INFER.get("proc") and INFER["proc"].poll() is None:
        workers.append({"kind": "infer", "pid": INFER["proc"].pid, "rss": proc_rss(INFER["proc"].pid)})
    return {"server_rss_mb": rss, "workers": workers,
            "mem_total_mb": mem.get("MemTotal", 0), "mem_available_mb": mem.get("MemAvailable", 0),
            "swap_total_mb": mem.get("SwapTotal", 0),
            "swap_used_mb": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)}


def proc_rss(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except (OSError, IndexError):
        return 0


def announce_and_open(url, also_lan=""):
    """等 server 真的起來再印網址並開瀏覽器——太早開會看到連線失敗。"""
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    bar = "═" * 52
    lines = [f"\n{bar}", "  DefectMaster Lite 已啟動", "", f"    {url}"]
    if also_lan:
        lines.append(f"    {also_lan}   （區網其他裝置）")
    lines += ["", "  按 Ctrl+C 結束", bar + "\n"]
    print("\n".join(lines), flush=True)

    if os.environ.get("LITE_NO_BROWSER") or not CONFIG.get("open_browser", True):
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log("無桌面環境（DISPLAY 未設定），略過自動開啟瀏覽器")
        return
    try:
        import webbrowser
        webbrowser.open(url)
        log(f"已在瀏覽器開啟 {url}")
    except Exception as e:
        log(f"無法自動開啟瀏覽器（請手動點上面的網址）：{e}", "WARNING")


if __name__ == "__main__":
    import uvicorn
    host = CONFIG.get("host", "127.0.0.1")
    port = int(CONFIG.get("port", 8002))
    log(f"DefectMaster Lite 啟動：{host}:{port} weights={WEIGHTS_DIR} output={OUTPUT_DIR}")
    log(f"授權：{LICENSE_MSG}（MAC {MACHINE_MAC}）")

    url = f"http://127.0.0.1:{port}"
    lan = ""
    if host == "0.0.0.0":                      # 有開放區網才顯示區網網址
        try:
            import socket
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.connect(("8.8.8.8", 80))
            lan = f"http://{sk.getsockname()[0]}:{port}"
            sk.close()
        except Exception:
            lan = ""
    threading.Thread(target=announce_and_open, args=(url, lan), daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
