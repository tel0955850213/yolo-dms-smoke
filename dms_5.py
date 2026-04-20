import cv2
import time
import os
import json
import requests
import struct
import serial
import threading
import numpy as np

# YOLO 與 MLX90640 相關套件
from ultralytics import YOLO
import adafruit_mlx90640
from adafruit_extended_bus import ExtendedI2C as I2C

# 雷達解析套件 (確保 tinyFrame.py 放在同一個資料夾)
from tinyFrame import TinyFrame

# ==========================================
# 參數設定區
# ==========================================
MODEL_PATH          = "/home/tel0955850213/Desktop/smoke_only_0408/best.pt"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1480445491991806094/m2x-orlNQurasWk6cZ-Jq7HJSCMhZ081NQr5q6QQUZLMjRsGOgUGV3z0jZWt4LEapSn0"

# 硬體設定
CAMERA_INDEX = 0
RADAR_PORT   = '/dev/ttyUSB0'
RADAR_BAUD   = 1382400
I2C_BUS_ID   = 7

# YOLO 門檻設定
CONF_THRESH    = 0.25
SMOKE_MIN_CONF = 0.25
IOU_OVERLAP    = 0.6

# 警報冷卻時間（秒）
COOLDOWN = {
    "eyeclose": 2.0,
    "phone":    3.0,
    "smoke":    3.0,
    "yawn":     5.0,
}

# ==========================================
# 全域變數
# ==========================================
current_heart_rate = 0.0
current_max_temp   = 0.0

sleep_start_time = 0
phone_start_time = 0
last_alarm_time  = {}

COLOR_MAP = {
    "eyeopen":  (0, 255, 0),
    "eyeclose": (0, 0, 255),
    "face":     (200, 200, 200),
    "phone":    (255, 0, 255),
    "smoke":    (0, 0, 200),
    "yawn":     (0, 255, 255),
}

# ==========================================
# 模組 1：熱影像監聽 (Log 追蹤 + 智慧鎖溫)
# ==========================================
def thermal_listen_thread():
    global current_max_temp
    print("🌡️ [熱影像] 正在啟動 I2C Bus 7...")
    try:
        i2c = I2C(I2C_BUS_ID)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        print("🌡️ [熱影像] MLX90640 連線成功！")
        
        frame_data = [0] * 768
        human_temp_locked = 0.0  # 用來記住鎖定的體溫
        
        # ⚠️ 你可以根據 Log 印出的原始溫度，調整這個補償值
        OFFSET = 2.4 
        
        while True:
            try:
                mlx.getFrame(frame_data)
                raw_temp = max(frame_data)
                
                # 1. 在背景印出 Log，讓你知道感測器真實抓到的溫度
                print(f"👉 [Thermal Log] 原始最高溫: {raw_temp:.2f} °C")
                
                # 2. 判斷是否為人體範圍 (30.5度 ~ 35.5度)
                if 30.5 <= raw_temp <= 35.5:
                    compensated_temp = raw_temp + OFFSET
                    
                    # 智慧鎖定：只在第一次偵測，或是人靠更近(溫度變高)時才更新鎖定值
                    # 這樣可以防止人頭稍微晃動導致溫度亂跳
                    if human_temp_locked == 0.0 or compensated_temp > human_temp_locked:
                        human_temp_locked = compensated_temp
                        print(f"🔒 [Thermal] 鎖定/更新人體溫度為: {human_temp_locked:.1f} °C")
                        
                    current_max_temp = human_temp_locked
                
                # 3. 判斷是否為非人體 (熱水、冰水，或人離開畫面)
                else:
                    if human_temp_locked != 0.0:
                        print("🔓 [Thermal] 偵測到非人體溫度，自動解除鎖定！")
                        human_temp_locked = 0.0 # 清除鎖定
                        
                    current_max_temp = raw_temp # 直接顯示熱水或環境的真實溫度
                    
            except ValueError:
                continue 
            except Exception as e:
                time.sleep(1)
    except Exception as e:
        print(f"❌ [熱影像] 無法初始化: {e}")

# ==========================================
# 模組 2：生命探測雷達監聽
# ==========================================
def radar_listen_thread():
    global current_heart_rate
    tf = TinyFrame()
    try:
        ser = serial.Serial(RADAR_PORT, RADAR_BAUD, timeout=1)
        print(f"📡 [雷達] 連線成功 ({RADAR_PORT})...")
    except Exception as e:
        print(f"❌ [雷達] 無法連接: {e}")
        return

    while True:
        try:
            raw_bytes = ser.read(ser.in_waiting or 1)
            for b in raw_bytes:
                tf.accept_byte(b)
                if tf.complete:
                    msg = tf.rf
                    if msg.type == 0x0A15 and len(msg.data) >= 4:
                        hr = struct.unpack('<f', msg.data[0:4])[0]
                        if hr > 0: 
                            current_heart_rate = hr
                    tf.complete = False
                    tf.reset_parser()
        except Exception:
            pass

# ==========================================
# 模組 3：Discord 雲端推播
# ==========================================
def send_discord_alert(alert_type, frame):
    global current_heart_rate, current_max_temp
    
    msgs = {
        "sleep": "🚨 **【DMS 警報】駕駛打瞌睡！** 閉眼過久，極度危險！",
        "phone": "⚠️ **【DMS 警報】駕駛分心！** 偵測到使用手機！",
        "yawn":  "🥱 **【DMS 提示】駕駛疲勞！** 偵測到打呵欠！",
        "smoke": "🚬 **【DMS 警報】駕駛抽菸！** 請注意安全！",
    }
    
    base_msg = msgs.get(alert_type, "🚨 【DMS 警報】偵測到異常行為！")
    
    full_msg = f"{base_msg}\n\n"
    full_msg += f"📊 **即時生理數據**：\n"
    full_msg += f"💓 心跳速率：**{current_heart_rate:.0f}** bpm\n"
    full_msg += f"🌡️ 駕駛體溫：**{current_max_temp:.1f}** °C"

    def _send():
        try:
            cv2.imwrite("dms_alert.jpg", frame)
            with open("dms_alert.jpg", "rb") as f:
                payload = {"content": full_msg}
                requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files={"file": f}, timeout=10)
            print(f"✅ [Discord] {alert_type} 警報發送成功 (體溫: {current_max_temp:.1f}C)")
        except Exception as e:
            print(f"❌ [Discord] 發送失敗: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ==========================================
# 輔助函式：IoU 計算與過濾
# ==========================================
def calc_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter)

def filter_boxes(all_boxes):
    filtered = [b for b in all_boxes if not (b["name"] == "smoke" and b["conf"] < SMOKE_MIN_CONF)]
    phone_boxes = [b["xyxy"] for b in filtered if b["name"] == "phone"]
    result = []
    for b in filtered:
        if b["name"] == "smoke" and any(calc_iou(b["xyxy"], p) > IOU_OVERLAP for p in phone_boxes):
            continue
        result.append(b)
    return result

# ==========================================
# 主程式：YOLO 視覺大腦
# ==========================================
def main():
    global sleep_start_time, phone_start_time, last_alarm_time, current_heart_rate, current_max_temp

    threading.Thread(target=thermal_listen_thread, daemon=True).start()
    threading.Thread(target=radar_listen_thread, daemon=True).start()

    print("🧠 載入 YOLO 模型...")
    model = YOLO(MODEL_PATH) 
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("🚀 超級 DMS 系統 v5 (智慧鎖溫版) 啟動！")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        now = time.time()
        results = model(frame, conf=CONF_THRESH, verbose=False)

        raw_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_boxes.append({
                "name": model.names[cls_id],
                "conf": float(box.conf[0]),
                "xyxy": box.xyxy[0].cpu().numpy().tolist()
            })

        boxes = filter_boxes(raw_boxes)
        detected = {b["name"] for b in boxes}

        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            color = COLOR_MAP.get(b["name"], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{b['name']} {b['conf']:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(frame, f"Temp: {current_max_temp:.1f}C | HR: {current_heart_rate:.0f}bpm", 
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if "eyeclose" in detected:
            if sleep_start_time == 0: sleep_start_time = now
            elif now - sleep_start_time >= 1.5:
                if now - last_alarm_time.get("sleep", 0) > COOLDOWN["eyeclose"]:
                    send_discord_alert("sleep", frame)
                    last_alarm_time["sleep"] = now
            cv2.putText(frame, f"Closing: {now - sleep_start_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            sleep_start_time = 0

        if "phone" in detected:
            if phone_start_time == 0: phone_start_time = now
            elif now - phone_start_time >= 1.0:
                if now - last_alarm_time.get("phone", 0) > COOLDOWN["phone"]:
                    send_discord_alert("phone", frame)
                    last_alarm_time["phone"] = now
            cv2.putText(frame, f"Phone: {now - phone_start_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            phone_start_time = 0

        if "yawn" in detected:
            if now - last_alarm_time.get("yawn", 0) > COOLDOWN["yawn"]:
                send_discord_alert("yawn", frame)
                last_alarm_time["yawn"] = now

        if "smoke" in detected:
            if now - last_alarm_time.get("smoke", 0) > COOLDOWN["smoke"]:
                send_discord_alert("smoke", frame)
                last_alarm_time["smoke"] = now

        # 因為你是無頭模式(沒接螢幕)，我已經幫你把 imshow 註解掉了，不會報錯閃退！
        # cv2.imshow("Super DMS System v5", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
