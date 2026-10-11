# DefectMaster Lite — 輕量版（Jetson）

與完整版 `app_for_offline.py` 相同的工作流程（資料集 → 標註 → 訓練 → 評估 → 推論），
但把記憶體佔用壓到最低。可與完整版**同時執行**（本版用 port 8002）。

## 啟動

```bash
cd "DefectMaster_Lite"
python3 server.py            # http://<jetson-ip>:8002
```

相依：`fastapi` / `uvicorn` / `pyyaml` / `pillow`（都已隨完整版安裝）。
`torch` / `ultralytics` 只有訓練與推論的子行程才會載入。

## 為什麼比較省

| | 完整版 | 輕量版 | 實測 |
|---|---|---|---|
| Server 常駐記憶體 | 312 MB（開機就 import torch） | **42–51 MB** | `/sysinfo` |
| 前端下載量 | 4.45 MB（React + Babel + Tailwind + lucide）| **40 KB**（純 JS/CSS）| 檔案大小 |
| 瀏覽器開頁 | 每次載入即時編譯 364 KB JSX | 無編譯步驟 | — |
| 訓練時 | torch 留在 server 行程內 | 子行程 1.9 GB，**結束即歸還** | 實測 |
| 推論時 | 模型常駐 | 子行程；閒置 180 秒自動結束 | 釋放後可用記憶體 +700 MB |
| 圖片牆 | 前端載原圖 | Server 端快取縮圖（61 KB → 6.6 KB）| 實測 |
| 資料集 | 整包 images + labelsMap 進瀏覽器 | Server 只存路徑，前端分頁取用 | — |

架構重點：**主 server 永遠不 import torch**。訓練走 `workers/train_worker.py`
子行程，推論走 `workers/infer_worker.py`（閒置自我了結），記憶體用完就真的還給系統。

## 訓練不複製資料集

- **data.yaml 模式**：直接把使用者的 yaml 餵給 Ultralytics，一張圖都不複製。
- **資料夾模式**：產生 `train.txt` / `val.txt`（絕對路徑清單）＋ `data.yaml`，
  以分層 8:2 拆分，同樣零複製。臨時檔放在 `DefectMaster_Output/.lite_jobs/`。

輸出沿用完整版的目錄慣例：訓練結果在 `DefectMaster_Output/<name>/`，
`best.pt` 複製為 `DefectMaster_Output/models/<name>.pt`，並寫入相容的 `metrics.json`
——所以**兩邊的模型與評估資產互通**。

## 標註

- 邊畫邊存：每張圖改完立刻寫成 YOLO `.txt`（`images/` → `labels/` 慣例；
  沒有 images/ 結構時就寫在圖片同層，Ultralytics 兩種都吃）。
- 快捷鍵：`←` / `→` 換圖、`1`–`9` 切類別、`Delete` 刪除選取框。
- 類別只允許**新增**與**改名**。刪除或調換順序會讓既有 `.txt` 的類別編號全部錯位，
  所以後端直接擋下。

## 已知範圍（刻意不做）

- 標註只支援 **BBox**。載入 Seg / OBB 資料集可以檢視（顯示外接框、標為唯讀）
  並照原 yaml 訓練，但不能在此編輯多邊形 → 用完整版。
- 不含 SAM 輔助標註、自動標註、分類（cls）資料集的 yaml 載入。
- 推論為單張；批次推論用完整版。

## 檔案

```
DefectMaster_Lite/
├── server.py           # FastAPI，無任何 ML import
├── licensing.py        # 與完整版同一套 MAC + HMAC 授權檢查
├── config.json         # port / 推論閒置秒數
├── workers/
│   ├── train_worker.py # 訓練子行程（唯一 import torch 的地方之一）
│   └── infer_worker.py # 推論子行程，閒置自動結束
└── web/                # index.html + style.css + app.js（無任何第三方 JS）
```
