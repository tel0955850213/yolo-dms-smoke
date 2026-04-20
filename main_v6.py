import os
import cv2
import time
import json
import struct
import pyaudio
import serial
import threading
import requests
import board
import busio
import adafruit_mlx90640
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from tinyFrame import TinyFrame

# ==========================================
# 參數設定區
# ==========================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1480445491991806094/m2x-orlNQurasWk6cZ-Jq7HJSCMhZ081NQr5q6QQUZLMjRsGOgUGV3z0jZWt4LEapSn0"
YOLO_MODEL_PATH = "/home/tel0955850213/Desktop/best.engine"
VOSK_MODEL_PATH = "../model"

RADAR_PORT = '/dev/ttyUSB0'
RADAR_BAUD = 1382400

COOLDOWN_TIME = 15  # 各類警報獨立的冷卻時間(秒)
CONFIRM_TIME = 2.5  # 連續跌倒幾秒才算數
TEMP_THRESHOLD = 31.0  # 判定為手掌靠近的溫度門檻

# --- 全域變數 ---
current_heart_rate = 0.0
current_breath_rate = 0.0
current_max_temp = 0.0

# 🌟 救援模式旗標
rescue_mode = False

# 🌟 獨立冷卻計時器
last_fall_alert = 0
last_thermal_alert = 0
last_voice_alert = 0
latest_frame = None

# ==========================================
# 模組 1：Discord 雲端推播
# ==========================================
def send_discord_alert(trigger_source, alert_type="normal"):
    global current_heart_rate, current_breath_rate, current_max_temp, latest_frame

    if alert_type == "interactive":
        msg = "🟢 **【狀態更新】傷者出現互動反應！** 意識清醒，持續發出熱感應求助訊號！"
    elif alert_type == "voice":
        msg = "🗣️ **【語音求救】聽到目標呼救聲！** 傷者意識清醒，請盡速支援！"
    else:
        msg = f"🚨 **【無人機搜救警報】** 觸發來源：{trigger_source}"

    message = f"{msg}\n"
    message += f"🌡️ 互動溫度: {current_max_temp:.1f} °C\n"
    message += f"💓 即時心跳: {current_heart_rate:.0f} bpm\n"
    message += f"🫁 即時呼吸: {current_breath_rate:.0f} 次/分\n"
    message += "請救援人員盡速確認現場狀態！"

    def _send():
        try:
            if latest_frame is not None:
                cv2.imwrite("alert_snapshot.jpg", latest_frame)
                with open("alert_snapshot.jpg", "rb") as f:
                    payload = {"content": message}
                    requests.post(DISCORD_WEBHOOK_URL, data={"payload_json": json.dumps(payload)}, files={"file": f}, timeout=10)
            else:
                requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=5)
            print(f"\n✅ [Discord] {trigger_source} 警報發送成功！")
        except Exception as e:
            print(f"\n❌ [Discord] 發送失敗: {e}")

    threading.Thread(target=_send).start()

# ==========================================
# 模組 1.5：熱影像監聽（MLX90640 I2C版本）
# ==========================================
def thermal_listen_thread():
    global current_max_temp
    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        print("🌡️ [熱影像] MLX90640 已連線...")
    except Exception as e:
        print(f"❌ [熱影像] 無法連接 MLX90640: {e}")
        return

    frame = [0] * 768  # 32x24 = 768 pixels
    while True:
        try:
            mlx.getFrame(frame)
            current_max_temp = max(frame)
        except Exception:
            pass

# ==========================================
# 模組 2：心跳雷達監聽
# ==========================================
def radar_listen_thread():
    global current_heart_rate, current_breath_rate
    tf = TinyFrame()

    try:
        ser = serial.Serial(RADAR_PORT, RADAR_BAUD, timeout=1)
        print("📡 [雷達] 生命探測雷達已連線...")
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
                if msg.type == 0x0A14 and len(msg.data) >= 4:
                    br = struct.unpack('<f', msg.data[:4])[0]
                    if br > 0:
                        current_breath_rate = br
                elif msg.type == 0x0A15 and len(msg.data) >= 4:
                    hr = struct.unpack('<f', msg.data[:4])[0]
                    if hr > 0:
                        current_heart_rate = hr
                tf.complete = False
                tf.reset_parser()
        except Exception:
            pass

# ==========================================
# 模組 3：Vosk 離線語音
# ==========================================
def vosk_listen_thread():
    global last_voice_alert, rescue_mode
    if not os.path.exists(VOSK_MODEL_PATH):
        return
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()

    mic_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '')
        if ("USB" in name or "Huawei" in name or "KT" in name) and info.get('maxInputChannels') > 0:
            mic_index = i
            break

    if mic_index is None:
        return

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                    input_device_index=mic_index, frames_per_buffer=8000)
    stream.start_stream()
    print("🎤 [語音] 麥克風已就緒...")

    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").replace(" ", "")
                if text and any(k in text for k in ["救命", "幫幫忙", "跌倒", "救我"]):
                    if time.time() - last_voice_alert > COOLDOWN_TIME:
                        print("🗣️ [語音觸發] 聽到呼救聲！")
                        send_discord_alert("視覺盲區-語音求救", alert_type="voice")
                        last_voice_alert = time.time()
                        rescue_mode = True
        except Exception:
            pass

# ==========================================
# 模組 4：主程式
# ==========================================
def main():
    global last_fall_alert, last_thermal_alert, rescue_mode, latest_frame

    threading.Thread(target=radar_listen_thread, daemon=True).start()
    threading.Thread(target=vosk_listen_thread, daemon=True).start()
    threading.Thread(target=thermal_listen_thread, daemon=True).start()

    print("👁️ [視覺] 正在載入 TensorRT 引擎...")
    model = YOLO(YOLO_MODEL_PATH, task="detect")
    cap = cv2.VideoCapture(0)

    is_falling = False
    fall_start_time = 0

    print("🚀 [系統] 多感測器無人機搜救大腦啟動完成！")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5)

        current_frame_has_fall = False
        for box in results[0].boxes:
            if model.names[int(box.cls[0])] == "fall":
                current_frame_has_fall = True
                break

        annotated_frame = results[0].plot()
        latest_frame = annotated_frame

        if current_frame_has_fall:
            if not is_falling:
                is_falling = True
                fall_start_time = time.time()
            else:
                if time.time() - fall_start_time >= CONFIRM_TIME:
                    if not rescue_mode:
                        if time.time() - last_fall_alert > COOLDOWN_TIME:
                            print("\n🚨 [視覺觸發] 確認跌倒！進入【持續救援模式】...")
                            send_discord_alert("視覺確認目標跌倒", alert_type="normal")
                            last_fall_alert = time.time()
                            rescue_mode = True
        else:
            is_falling = False
            fall_start_time = 0

        if rescue_mode:
            if current_max_temp >= TEMP_THRESHOLD:
                if time.time() - last_thermal_alert > COOLDOWN_TIME:
                    print(f"\n🟢 [熱感應互動] 傷者摸了感測器 ({current_max_temp}C)！回報清醒狀態！")
                    send_discord_alert("救援模式-傷者主動觸碰", alert_type="interactive")
                    last_thermal_alert = time.time()

        cv2.putText(annotated_frame, f"T:{current_max_temp:.0f}C | HR:{current_heart_rate:.0f} | BR:{current_breath_rate:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if is_falling and not rescue_mode:
            remain = max(0, CONFIRM_TIME - (time.time() - fall_start_time))
            cv2.putText(annotated_frame, f"Confirming: {remain:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if rescue_mode:
            cv2.putText(annotated_frame, "[RESCUE MODE ACTIVE]", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Touch sensor to respond!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Drone Rescue System v6", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            print("🔄 [系統] 手動解除救援模式，重新開始巡邏。")
            rescue_mode = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
