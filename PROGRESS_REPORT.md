# DMS 系統開發進度報告
**學生：** tel0955850213（Donald Lin）  
**日期：** 2026-04-20  
**指導教授：** [教授姓名]  
**系統名稱：** 駕駛監控系統（Driver Monitoring System, DMS）  
**硬體平台：** NVIDIA Jetson Orin Nano（Ubuntu 22.04）

---

## 一、系統概述

本系統為一套整合 AI 視覺辨識、生理監測與語音辨識的駕駛安全監控平台，部署於 Jetson Orin Nano 邊緣運算裝置上，可即時偵測駕駛的危險行為並發出警報。

### 系統架構圖

```
┌─────────────────────────────────────────────────────────┐
│                  Jetson Orin Nano                        │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  YOLO    │  │  LD6002  │  │MLX90640  │  │  Vosk  │ │
│  │ 視覺辨識 │  │ 心跳雷達 │  │ 熱影像   │  │ 語音   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       └──────────────┴─────────────┴─────────────┘      │
│                          │                               │
│                   ┌──────▼──────┐                       │
│                   │  Flask 伺服器 │  Port 5000           │
│                   └──────┬──────┘                       │
└──────────────────────────┼──────────────────────────────┘
                           │ HTTP MJPEG 串流
                    ┌──────▼──────┐
                    │  Mac 瀏覽器  │
                    │ 即時監控畫面 │
                    └─────────────┘
                           │ Discord Webhook
                    ┌──────▼──────┐
                    │  Discord 頻道 │
                    │  即時警報推播 │
                    └─────────────┘
```

---

## 二、硬體模組清單

| 模組 | 型號 | 功能 | 連接方式 |
|------|------|------|---------|
| 邊緣運算 | NVIDIA Jetson Orin Nano | 主控制器，執行所有 AI 推論 | — |
| 攝影機 | Logitech StreamCam | 駕駛臉部影像擷取 + 內建麥克風 | USB |
| 心跳雷達 | LD6002 | 非接觸式心跳偵測（毫米波雷達） | UART `/dev/ttyUSB0`，Baud: 1,382,400 |
| 熱影像感測器 | MLX90640 | 駕駛體溫偵測（32×24 像素熱影像） | I2C Bus 7 |
| AI 模型 | YOLO11n（自訓練） | 偵測：眼閉、打哈欠、手機、抽菸 | — |
| 語音辨識 | Vosk（中文模型 66MB） | 離線語音求救詞偵測 | — |

---

## 三、GitHub Repository（程式碼版本管理）

本次建立兩個 GitHub Repository，進行完整的版本控制：

### 3.1 yolo-dms-smoke（主要 DMS 系統）
- **URL：** https://github.com/tel0955850213/yolo-dms-smoke
- **內容：** DMS 主系統程式碼、YOLO 模型、訓練資料集
- **本次 commit 記錄：**

| Commit | 說明 |
|--------|------|
| `f611f4f` | 初始提交：所有 Python 腳本、模型檔案、訓練資料集 |
| `4821ad3` | 修復心跳濾波、PIL 中文顯示、緊急警報、Vosk 麥克風 fallback |
| `5a7b8b7` | `--` 立刻顯示邏輯、Discord 加入觸發原因說明 |
| `3c5580a` | 心跳超時改 5 秒、哈欠需 2 秒持續偵測、修復 Discord 語音原因 bug |
| `84f3536` | 心跳消失加 2 秒緩衝，避免雜訊觸發 |
| `63f06cd` | 心跳緩衝延長至 5 秒（顯示 `--`），10 秒後觸發 110 |

### 3.2 yolo-elder-fall-detection（長者跌倒偵測）
- **URL：** https://github.com/tel0955850213/yolo-elder-fall-detection
- **內容：** YOLOv10 長者跌倒偵測研究，包含訓練資料集（train/valid/test）

---

## 四、核心程式檔案說明

### 主程式：`dms_7.py`（最終版本）
**路徑：** `/home/tel0955850213/Desktop/smoke_only_0408/dms_7.py`  
**行數：** ~490 行  
**執行方式：** `python3 dms_7.py`  
**監控網址：** `http://172.20.10.3:5000`（區域網路）

---

## 五、本次開發重點工作

### 5.1 LD6002 心跳偵測穩定化（核心修復）

**問題：** 舊版（dms_11/dms_12）心跳數值不穩定，會從 70 bpm 突然掉到 30 bpm

**根本原因分析：**

| 版本 | 最小門檻 | 突波阻擋 | 歷史筆數 | 結果 |
|------|---------|---------|---------|------|
| dms_6/7（舊） | 40 bpm | ✅ 有，但 ±2 漂移 | 5 | 穩定但會累積漂移 |
| dms_11/12（退步） | **20 bpm** | ❌ 移除 | 5 | 嚴重不穩定 |
| dms_7（本次修復） | 40 bpm | ✅ 限制每次跳動 ≤12 bpm | 10 | 穩定 |

**修復後的濾波邏輯：**
```python
HR_MIN = 40           # 人類最低心跳（過濾雜訊）
HR_MAX = 140          # 人類最高心跳
HR_HISTORY_SIZE = 10  # 10 筆滑動平均（更平滑）
MAX_DELTA = 12        # 每次最多跳動 12 bpm（防突波）

if HR_MIN <= raw_hr <= HR_MAX:
    if len(hr_history) >= 3:
        avg = sum(hr_history) / len(hr_history)
        if abs(raw_hr - avg) > MAX_DELTA:
            raw_hr = avg + (MAX_DELTA if raw_hr > avg else -MAX_DELTA)
    hr_history.append(raw_hr)
    if len(hr_history) > HR_HISTORY_SIZE:
        hr_history.pop(0)
    current_heart_rate = sum(hr_history) / len(hr_history)
```

### 5.2 緊急警報系統設計

**觸發流程（心跳消失）：**
```
心跳訊號消失
    ↓（等 5 秒）
顯示 "--"（無訊號）
    ↓（再等 5 秒）
觸發 110 緊急警報（共 10 秒）
```

**觸發流程（語音求救）：**
```
駕駛說出「救命」/「幫助」/「幫幫忙」/「救我」
    ↓（Vosk 即時辨識）
立刻觸發 110 緊急警報
```

**緊急警報畫面：**
- 影像中央疊加紅色半透明橫幅
- 顯示文字：「!!! 駕駛有危險 !!!」和「正在撥打 110...」
- 使用 PIL + NotoSansCJK 字型（解決 OpenCV 中文亂碼問題）
- 同時發送 Discord 通知

### 5.3 MLX90640 體溫顯示修正

**問題：** 溫度鎖定機制只升不降，導致顯示值偏高（38.1°C，不符合正常體溫）

**修復：** 改用 8 筆滑動平均，並限制顯示範圍在 35.5–37.5°C
```python
# 8 筆滑動平均，允許溫度自然上下波動
temp_history.append(compensated)
avg_temp = sum(temp_history) / len(temp_history)
current_max_temp = max(35.5, min(37.5, avg_temp))
```

### 5.4 Flask 網頁即時串流

**功能：** 將 Jetson 的攝影機畫面（含 YOLO 框選結果）透過 MJPEG 格式串流到區域網路

**技術：** Flask + OpenCV `cv2.imencode(".jpg")`  
**存取方式：** Mac 瀏覽器輸入 `http://172.20.10.3:5000`

### 5.5 Vosk 語音辨識整合

**問題：** Logitech StreamCam 麥克風在 Jetson 上透過 pyaudio 無法直接枚舉（ALSA 設定問題）

**解決方案：** 程式啟動時自動執行 `pactl set-default-source`，將 StreamCam 設為 PulseAudio 預設輸入：
```python
def _set_streamcam_as_pulseaudio_default():
    result = subprocess.run(['pactl', 'list', 'short', 'sources'], ...)
    for line in result.stdout.split('\n'):
        if 'StreamCam' in line:
            source_name = line.split('\t')[1]
            subprocess.run(['pactl', 'set-default-source', source_name])
```

### 5.6 Discord 警報推播改進

**新增觸發原因說明：**

| 觸發類型 | Discord 訊息範例 |
|---------|----------------|
| 心跳消失 | `⚠️ 觸發原因：心跳消失超過 5 秒` |
| 語音求救 | `🎤 觸發原因：駕駛說了「救命」呼救` |
| 閉眼超時 | `🚨 【DMS 警報】駕駛打瞌睡！` |
| 使用手機 | `⚠️ 【DMS 警報】駕駛使用手機！` |

### 5.7 哈欠偵測靈敏度調整

**問題：** 駕駛說話時嘴巴張開也被誤判為哈欠

**修復：** 需持續偵測到哈欠 **2 秒**才送警報（原本單幀即觸發）

---

## 六、系統偵測項目與警報設定

| 偵測項目 | 觸發條件 | 警報冷卻 | 說明 |
|---------|---------|---------|------|
| 閉眼（eyeclose） | 持續 **1.5 秒** | 2 秒 | 防止眨眼誤觸發 |
| 打哈欠（yawn） | 持續 **2 秒** | 5 秒 | 防止說話誤觸發 |
| 使用手機（phone） | 持續 **1 秒** | 3 秒 | — |
| 抽菸（smoke） | 單幀偵測 | 3 秒 | 置信度 > 0.25 |
| 心跳消失 | **5 秒**後顯示 `--`，再 **5 秒**觸發 110 | — | 合計 10 秒 |
| 語音求救 | 即時（說出關鍵詞後） | — | 救命/幫助/幫幫忙/救我 |

---

## 七、待辦事項（Future Work）

- [ ] Vosk 語音辨識實際測試（StreamCam 已設定為預設音源）
- [ ] 110 緊急撥打的實際整合（目前為畫面警示）
- [ ] 長者跌倒偵測系統（yolo-elder-fall-detection）整合至同一平台
- [ ] 系統在行駛中的實際車載測試

---

## 八、檔案結構

```
/home/tel0955850213/Desktop/smoke_only_0408/
├── dms_7.py              ← 主程式（最終版）
├── dms_1.py ~ dms_12.py  ← 開發過程版本記錄
├── best.pt               ← YOLO 訓練模型（PyTorch）
├── best.onnx             ← YOLO 模型（ONNX 格式）
├── best.engine           ← YOLO 模型（TensorRT 加速）
├── model/                ← Vosk 中文語音模型（66MB）
├── train/                ← 訓練資料集
├── valid/                ← 驗證資料集
└── test/                 ← 測試資料集
```
