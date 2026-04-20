import cv2
import time
import os
import json
import requests
import struct
import serial
import threading
import numpy as np
from flask import Flask, Response
from PIL import ImageFont, ImageDraw, Image

from ultralytics import YOLO
import adafruit_mlx90640
from adafruit_extended_bus import ExtendedI2C as I2C
from tinyFrame import TinyFrame

try:
    import pyaudio
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ==========================================
# 參數設定區
# ==========================================
MODEL_PATH          = "/home/tel0955850213/Desktop/smoke_only_0408/best.pt"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1480445491991806094/m2x-orlNQurasWk6cZ-Jq7HJSCMhZ081NQr5q6QQUZLMjRsGOgUGV3z0jZWt4LEapSn0"
VOSK_MODEL_PATH     = "/home/tel0955850213/Desktop/smoke_only_0408/model"
CHINESE_FONT_PATH   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

CAMERA_INDEX = 0
RADAR_PORT   = '/dev/ttyUSB0'
RADAR_BAUD   = 1382400
I2C_BUS_ID   = 7

CONF_THRESH    = 0.25
SMOKE_MIN_CONF = 0.25
IOU_OVERLAP    = 0.6

COOLDOWN = {
    "eyeclose": 2.0,
    "phone":    3.0,
    "smoke":    3.0,
    "yawn":     5.0,
}

# LD6002 心跳濾波參數
HR_MIN          = 40
HR_MAX          = 140
HR_HISTORY_SIZE = 10
MAX_DELTA       = 12
HR_TIMEOUT_SEC  = 5.0

# ==========================================
# 全域變數
# ==========================================
current_heart_rate  = 0.0
hr_history          = []
last_valid_hr_time  = 0.0
hr_lost_time        = 0.0   # 心跳訊號消失的時間（用來計算 10 秒後觸發 110）
emergency_reason    = ""    # 觸發原因：'heartbeat' 或 'voice'

current_max_temp    = 0.0

emergency_mode      = False
emergency_lock      = threading.Lock()

sleep_start_time = 0
phone_start_time = 0
yawn_start_time  = 0
last_alarm_time  = {}

output_frame = None
frame_lock   = threading.Lock()

app = Flask(__name__)

COLOR_MAP = {
    "eyeopen":  (0, 255, 0),
    "eyeclose": (0, 0, 255),
    "face":     (200, 200, 200),
    "phone":    (255, 0, 255),
    "smoke":    (0, 0, 200),
    "yawn":     (0, 255, 255),
}

# ==========================================
# 輔助：PIL 中文字渲染（解決 cv2 中文亂碼）
# ==========================================
_font_cache = {}

def put_chinese_text(frame, text, pos, font_size=42, color=(255, 255, 255)):
    if font_size not in _font_cache:
        _font_cache[font_size] = ImageFont.truetype(CHINESE_FONT_PATH, font_size)
    font    = _font_cache[font_size]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 模組 1：MLX90640 熱影像監聽（滑動平均溫度）
# ==========================================
def thermal_listen_thread():
    global current_max_temp
    print("🌡️ [MLX90640] 正在啟動 I2C Bus 7...")
    try:
        i2c = I2C(I2C_BUS_ID)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        print("🌡️ [MLX90640] 連線成功！")

        frame_data    = [0] * 768
        temp_history  = []
        TEMP_HIST_SIZE = 8
        OFFSET         = 1.2

        while True:
            try:
                mlx.getFrame(frame_data)
                img_temp   = np.array(frame_data).reshape((24, 32))
                driver_roi = img_temp[6:18, 10:22]
                raw_temp   = np.max(driver_roi)

                # 只接受人體範圍（30.5–38.0）
                if 30.5 <= raw_temp <= 38.0:
                    compensated = raw_temp + OFFSET if raw_temp <= 35.5 else raw_temp

                    # 滑動平均（允許溫度自然上下波動）
                    temp_history.append(compensated)
                    if len(temp_history) > TEMP_HIST_SIZE:
                        temp_history.pop(0)

                    avg_temp = sum(temp_history) / len(temp_history)

                    # 正常人體溫度範圍限制在 35.5–37.5°C
                    current_max_temp = max(35.5, min(37.5, avg_temp))
                else:
                    temp_history.clear()
                    current_max_temp = raw_temp

            except ValueError:
                continue
            except Exception:
                time.sleep(1)
    except Exception as e:
        print(f"❌ [MLX90640] 無法初始化: {e}")

# ==========================================
# 模組 2：LD6002 雷達心跳監聽（穩定濾波版）
# ==========================================
def radar_listen_thread():
    global current_heart_rate, hr_history, last_valid_hr_time, hr_lost_time
    global emergency_mode, emergency_reason
    tf = TinyFrame()
    try:
        ser = serial.Serial(RADAR_PORT, RADAR_BAUD, timeout=1)
        print(f"📡 [LD6002] 連線成功 ({RADAR_PORT})...")
    except Exception as e:
        print(f"❌ [LD6002] 無法連接: {e}")
        return

    while True:
        try:
            raw_bytes = ser.read(ser.in_waiting or 1)
            for b in raw_bytes:
                tf.accept_byte(b)
                if tf.complete:
                    msg = tf.rf
                    if msg.type == 0x0A15 and len(msg.data) >= 4:
                        raw_hr = struct.unpack('<f', msg.data[0:4])[0]

                        if HR_MIN <= raw_hr <= HR_MAX:
                            # 心跳恢復：重置所有計時器
                            last_valid_hr_time = time.time()
                            hr_lost_time = 0.0

                            if len(hr_history) >= 3:
                                avg = sum(hr_history) / len(hr_history)
                                if abs(raw_hr - avg) > MAX_DELTA:
                                    raw_hr = avg + (MAX_DELTA if raw_hr > avg else -MAX_DELTA)

                            hr_history.append(raw_hr)
                            if len(hr_history) > HR_HISTORY_SIZE:
                                hr_history.pop(0)

                            current_heart_rate = sum(hr_history) / len(hr_history)

                            with emergency_lock:
                                if emergency_mode:
                                    emergency_mode = False
                                    emergency_reason = ""
                                    print("✅ [LD6002] 心跳恢復，解除緊急模式")

                        else:
                            now = time.time()
                            # 步驟 1：記錄第一次收不到合法心跳的時間
                            if last_valid_hr_time > 0 and hr_lost_time == 0.0:
                                hr_lost_time = now

                            # 步驟 2：超過 2 秒才顯示 '--'（過濾單一雜訊）
                            if hr_lost_time > 0 and now - hr_lost_time > 5.0 and current_heart_rate > 0:
                                hr_history.clear()
                                current_heart_rate = 0.0
                                print("⚠️  [LD6002] 心跳訊號消失 2 秒，顯示 '--'")

                            # 步驟 3：顯示 '--' 後再等 HR_TIMEOUT_SEC 秒才觸發 110
                            if hr_lost_time > 0 and now - hr_lost_time > 5.0 + HR_TIMEOUT_SEC:
                                with emergency_lock:
                                    if not emergency_mode:
                                        emergency_mode = True
                                        emergency_reason = "heartbeat"
                                        print(f"🚨 [LD6002] 心跳消失超過 {HR_TIMEOUT_SEC:.0f} 秒，觸發緊急警報！")

                    tf.complete = False
                    tf.reset_parser()
        except Exception:
            pass

# ==========================================
# 模組 3：Vosk 語音監聽
# ==========================================
def _set_streamcam_as_pulseaudio_default():
    import subprocess
    try:
        result = subprocess.run(['pactl', 'list', 'short', 'sources'],
                                capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'StreamCam' in line or 'streamcam' in line.lower():
                source_name = line.split('\t')[1]
                subprocess.run(['pactl', 'set-default-source', source_name], check=True)
                print(f"🎤 [Vosk] StreamCam 設為 PulseAudio 預設音源: {source_name}")
                return True
    except Exception as e:
        print(f"⚠️ [Vosk] 無法設定 PulseAudio 預設音源: {e}")
    return False


def vosk_listen_thread():
    global emergency_mode, emergency_reason
    if not VOSK_AVAILABLE or not os.path.exists(VOSK_MODEL_PATH):
        print("⚠️ [Vosk] 模型未找到，語音功能略過")
        return

    # 先把 StreamCam 設為 PulseAudio 預設輸入（讓 pulse 裝置走 StreamCam）
    _set_streamcam_as_pulseaudio_default()

    model = Model(VOSK_MODEL_PATH)
    rec   = KaldiRecognizer(model, 16000)
    p     = pyaudio.PyAudio()

    # 找 pulse 裝置（已設定為 StreamCam）
    mic_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '').lower()
        if info.get('maxInputChannels') > 0 and 'pulse' in name:
            mic_index = i
            print(f"🎤 [Vosk] 使用 PulseAudio → StreamCam (index {i})")
            break

    # Fallback：default
    if mic_index is None:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0 and 'default' in info.get('name', '').lower():
                mic_index = i
                print(f"🎤 [Vosk] 使用系統預設裝置 (index {i})")
                break

    if mic_index is None:
        print("⚠️ [Vosk] 找不到任何可用麥克風，語音功能略過")
        return

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                    input_device_index=mic_index, frames_per_buffer=8000)
    stream.start_stream()
    print("🎤 [Vosk] 麥克風就緒，監聽：救命 / 幫助 / 幫幫忙 / 救我")

    TRIGGER_WORDS = ["救命", "幫助", "幫幫忙", "救我"]

    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text   = result.get("text", "").replace(" ", "")
                if text and any(k in text for k in TRIGGER_WORDS):
                    print(f"🎤 [Vosk] 偵測到求救詞：{text}")
                    with emergency_lock:
                        emergency_mode = True
                        emergency_reason = f"voice:{text}"
        except Exception:
            pass

# ==========================================
# 模組 4：Discord 推播
# ==========================================
def send_discord_alert(alert_type, frame, reason=""):
    global current_heart_rate, current_max_temp

    msgs = {
        "sleep":     "🚨 **【DMS 警報】駕駛打瞌睡！**",
        "phone":     "⚠️ **【DMS 警報】駕駛使用手機！**",
        "yawn":      "🥱 **【DMS 提示】駕駛疲勞！**",
        "smoke":     "🚬 **【DMS 警報】駕駛抽菸！**",
        "emergency": "🆘 **【緊急警報】駕駛有危險！正在撥打 110！**",
    }

    full_msg = msgs.get(alert_type, "🚨 異常！") + "\n\n"

    if alert_type == "emergency":
        if reason == "heartbeat":
            full_msg += f"⚠️ **觸發原因：心跳消失超過 {HR_TIMEOUT_SEC:.0f} 秒**\n\n"
        elif reason.startswith("voice:"):
            word = reason.split(":", 1)[1]
            full_msg += f"🎤 **觸發原因：駕駛說了「{word}」呼救**\n\n"

    hr_display = f"{current_heart_rate:.0f}" if current_heart_rate > 0 else "--"
    full_msg += f"💓 心跳：**{hr_display}** bpm\n"
    full_msg += f"🌡️ 體溫：**{current_max_temp:.1f}** °C"

    def _send():
        try:
            cv2.imwrite("dms_alert.jpg", frame)
            with open("dms_alert.jpg", "rb") as f:
                payload = {"content": full_msg}
                requests.post(DISCORD_WEBHOOK_URL,
                              data={"payload_json": json.dumps(payload)},
                              files={"file": f}, timeout=10)
        except Exception as e:
            print(f"❌ [Discord] 發送失敗: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ==========================================
# 輔助：IoU 計算
# ==========================================
def calc_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)

def filter_boxes(all_boxes):
    filtered    = [b for b in all_boxes if not (b["name"] == "smoke" and b["conf"] < SMOKE_MIN_CONF)]
    phone_boxes = [b["xyxy"] for b in filtered if b["name"] == "phone"]
    result = []
    for b in filtered:
        if b["name"] == "smoke" and any(calc_iou(b["xyxy"], p) > IOU_OVERLAP for p in phone_boxes):
            continue
        result.append(b)
    return result

# ==========================================
# 模組 5：緊急橫幅（中文 PIL 版）
# ==========================================
def draw_emergency_overlay(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 140), (640, 340), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    frame = put_chinese_text(frame, "!!! 駕駛有危險 !!!",
                             (55, 158), font_size=46, color=(255, 255, 255))
    frame = put_chinese_text(frame, "正在撥打 110...",
                             (105, 258), font_size=40, color=(0, 255, 255))
    return frame

# ==========================================
# 模組 6：YOLO 主迴圈
# ==========================================
def yolo_inference_loop():
    global sleep_start_time, phone_start_time, last_alarm_time
    global output_frame, frame_lock, emergency_mode

    emergency_discord_sent = False

    print("🧠 載入 YOLO 模型...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("🚀 DMS 系統 v7（穩定心跳 + 中文顯示 + 緊急警報）啟動！")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now     = time.time()
        results = model(frame, conf=CONF_THRESH, verbose=False)

        raw_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_boxes.append({
                "name": model.names[cls_id],
                "conf": float(box.conf[0]),
                "xyxy": box.xyxy[0].cpu().numpy().tolist(),
            })

        boxes    = filter_boxes(raw_boxes)
        detected = {b["name"] for b in boxes}

        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            color = COLOR_MAP.get(b["name"], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{b['name']} {b['conf']:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 狀態列（英文數字用 cv2，不需要 PIL）
        hr_display = f"{current_heart_rate:.0f}" if current_heart_rate > 0 else "--"
        cv2.putText(frame, f"Temp: {current_max_temp:.1f}C | HR: {hr_display} bpm",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 眼睛閉合
        if "eyeclose" in detected:
            if sleep_start_time == 0:
                sleep_start_time = now
            elif now - sleep_start_time >= 1.5:
                if now - last_alarm_time.get("sleep", 0) > COOLDOWN["eyeclose"]:
                    send_discord_alert("sleep", frame)
                    last_alarm_time["sleep"] = now
            cv2.putText(frame, f"Closing: {now - sleep_start_time:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            sleep_start_time = 0

        # 手機
        if "phone" in detected:
            if phone_start_time == 0:
                phone_start_time = now
            elif now - phone_start_time >= 1.0:
                if now - last_alarm_time.get("phone", 0) > COOLDOWN["phone"]:
                    send_discord_alert("phone", frame)
                    last_alarm_time["phone"] = now
            cv2.putText(frame, f"Phone: {now - phone_start_time:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            phone_start_time = 0

        if "yawn" in detected:
            if yawn_start_time == 0:
                yawn_start_time = now
            elif now - yawn_start_time >= 2.0:
                if now - last_alarm_time.get("yawn", 0) > COOLDOWN["yawn"]:
                    send_discord_alert("yawn", frame)
                    last_alarm_time["yawn"] = now
        else:
            yawn_start_time = 0

        if "smoke" in detected:
            if now - last_alarm_time.get("smoke", 0) > COOLDOWN["smoke"]:
                send_discord_alert("smoke", frame)
                last_alarm_time["smoke"] = now

        # 緊急模式
        with emergency_lock:
            is_emergency = emergency_mode

        if is_emergency:
            frame = draw_emergency_overlay(frame)
            if not emergency_discord_sent:
                with emergency_lock:
                    reason = emergency_reason
                send_discord_alert("emergency", frame, reason=reason)
                emergency_discord_sent = True
        else:
            emergency_discord_sent = False

        with frame_lock:
            output_frame = frame.copy()

    cap.release()

# ==========================================
# Flask 網頁串流
# ==========================================
def generate_frames():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            flag, encoded = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + bytearray(encoded) + b'\r\n')

@app.route("/")
def index():
    return """
    <html>
        <head><title>DMS v7 Live</title></head>
        <body style="background:#111;color:#fff;text-align:center;padding-top:40px;">
            <h2>Jetson Orin Nano — DMS 系統即時影像 v7</h2>
            <img src="/video_feed"
                 style="border:2px solid cyan;border-radius:8px;width:640px;height:480px;">
        </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ==========================================
# 主程式進入點
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=thermal_listen_thread, daemon=True).start()
    threading.Thread(target=radar_listen_thread,   daemon=True).start()
    threading.Thread(target=vosk_listen_thread,    daemon=True).start()
    threading.Thread(target=yolo_inference_loop,   daemon=True).start()

    print("🌐 [Flask] 即時影像伺服器啟動！")
    print("👉 Mac 瀏覽器輸入：http://172.20.10.3:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
