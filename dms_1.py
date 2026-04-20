import cv2
import time
import os
import json
import requests
import threading
from ultralytics import YOLO
from datetime import datetime

# ==========================================
# 參數設定區
# ==========================================
MODEL_PATH          = r"C:\Users\Lin\Desktop\CLAUDE_CRAZY\smoke_only_0408用了老半天抽菸不準確_yolov10資料夾的模型finetune再學抽菸\best.pt"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1480445491991806094/m2x-orlNQurasWk6cZ-Jq7HJSCMhZ081NQr5q6QQUZLMjRsGOgUGV3z0jZWt4LEapSn0"
CAMERA_INDEX        = 0
CONF_THRESH         = 0.25   # 整體偵測門檻
SMOKE_MIN_CONF      = 0.25   # smoke 最低信心值，低於此直接忽略
IOU_OVERLAP         = 0.6    # phone 與 smoke 框重疊超過此值 → 忽略 smoke（調高，避免過度過濾）

# 警報冷卻（秒）
COOLDOWN = {
    "eyeclose": 2.0,
    "phone":    3.0,
    "smoke":    3.0,
    "yawn":     5.0,
}

# 狀態追蹤
sleep_start_time  = 0
phone_start_time  = 0
last_alarm_time   = {}

# 顏色 (BGR)
COLOR_MAP = {
    "eyeopen":  (0, 255, 0),
    "eyeclose": (0, 0, 255),
    "face":     (200, 200, 200),
    "phone":    (255, 0, 255),
    "smoke":    (0, 0, 200),
    "yawn":     (0, 255, 255),
}

# ==========================================
# IoU 計算
# ==========================================
def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter)

# ==========================================
# 過濾誤判：phone 誤判成 smoke
# ==========================================
def filter_boxes(all_boxes):
    # 1. 先濾掉低信心 smoke
    filtered = []
    for b in all_boxes:
        if b["name"] == "smoke" and b["conf"] < SMOKE_MIN_CONF:
            print(f"[FILTER] smoke conf 太低 ({b['conf']:.2f}) → 忽略")
            continue
        filtered.append(b)

    # 2. 再過濾與 phone 框重疊的 smoke
    phone_boxes = [b["xyxy"] for b in filtered if b["name"] == "phone"]
    result = []
    for b in filtered:
        if b["name"] == "smoke":
            overlap = any(calc_iou(b["xyxy"], p) > IOU_OVERLAP for p in phone_boxes)
            if overlap:
                print(f"[FILTER] smoke 與 phone 重疊 → 忽略")
                continue
        result.append(b)
    return result

# ==========================================
# Discord 推播
# ==========================================
def send_discord_alert(alert_type, frame):
    msgs = {
        "sleep": "🚨 **【DMS 警報】駕駛打瞌睡！** 閉眼超過 1.5 秒，極度危險！",
        "phone": "⚠️ **【DMS 警報】駕駛分心！** 偵測到使用手機！",
        "yawn":  "🥱 **【DMS 提示】駕駛疲勞！** 偵測到打呵欠，建議停車休息。",
        "smoke": "🚬 **【DMS 警報】駕駛抽菸！** 請注意行車安全！",
    }
    msg = msgs.get(alert_type, "🚨 【DMS 警報】偵測到異常駕駛行為！")

    def _send():
        try:
            cv2.imwrite("dms_alert.jpg", frame)
            with open("dms_alert.jpg", "rb") as f:
                requests.post(
                    DISCORD_WEBHOOK_URL,
                    data={"payload_json": json.dumps({"content": msg})},
                    files={"file": f},
                    timeout=10
                )
            print(f"✅ [Discord] {alert_type} 發送成功")
        except Exception as e:
            print(f"❌ [Discord] 發送失敗: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ==========================================
# 主程式
# ==========================================
def main():
    global sleep_start_time, phone_start_time, last_alarm_time

    print("🧠 載入模型...")
    model = YOLO(MODEL_PATH)
    print("✅ Classes:", model.names)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = model(frame, conf=CONF_THRESH, verbose=False)
        now       = time.time()

        # 收集偵測框
        raw_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_boxes.append({
                "name": model.names[cls_id],
                "conf": float(box.conf[0]),
                "xyxy": box.xyxy[0].cpu().numpy().tolist()
            })

        # 過濾誤判
        boxes = filter_boxes(raw_boxes)

        # 當前偵測到的 class
        detected = {b["name"] for b in boxes}

        # ── 繪製框 ──────────────────────────────────
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            color = COLOR_MAP.get(b["name"], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{b['name']} {b['conf']:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── 閉眼計時 → 打瞌睡警報 ───────────────────
        if "eyeclose" in detected:
            if sleep_start_time == 0:
                sleep_start_time = now
            elif now - sleep_start_time >= 1.5:
                if now - last_alarm_time.get("sleep", 0) > COOLDOWN["eyeclose"]:
                    print("🚨 打瞌睡警報！")
                    send_discord_alert("sleep", frame)
                    last_alarm_time["sleep"] = now
            cv2.putText(frame, f"Closing: {now - sleep_start_time:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            sleep_start_time = 0

        # ── 手機計時 → 分心警報 ──────────────────────
        if "phone" in detected:
            if phone_start_time == 0:
                phone_start_time = now
            elif now - phone_start_time >= 1.0:
                if now - last_alarm_time.get("phone", 0) > COOLDOWN["phone"]:
                    print("⚠️ 手機使用警報！")
                    send_discord_alert("phone", frame)
                    last_alarm_time["phone"] = now
            cv2.putText(frame, f"Phone: {now - phone_start_time:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            phone_start_time = 0

        # ── 打呵欠 → 疲勞警報 ───────────────────────
        if "yawn" in detected:
            if now - last_alarm_time.get("yawn", 0) > COOLDOWN["yawn"]:
                print("🥱 打呵欠警報！")
                send_discord_alert("yawn", frame)
                last_alarm_time["yawn"] = now

        # ── 抽菸 → 警報 ─────────────────────────────
        if "smoke" in detected:
            if now - last_alarm_time.get("smoke", 0) > COOLDOWN["smoke"]:
                print("🚬 抽菸警報！")
                send_discord_alert("smoke", frame)
                last_alarm_time["smoke"] = now

        # ── 時間戳 ───────────────────────────────────
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("DMS - Driver Monitoring System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()