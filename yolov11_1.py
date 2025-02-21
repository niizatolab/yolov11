import cv2
from ultralytics import YOLO
from harvesters.core import Harvester
import numpy as np
import time

# GigEカメラのセットアップ
def setup_gige_camera():
    h = Harvester()
    h.add_file("C:/Program Files/YourCameraSDK/GenICam.xml")  # 適切なパスに変更
    h.update()
    ia = h.create_image_acquirer(0)
    ia.start()
    return ia

# YOLOv11モデルのロード
model = YOLO("yolo11n.pt")  # 適切なパスに変更

# カメラを初期化
ia = setup_gige_camera()
print("GigEカメラが接続されました。")

# OpenCVのウィンドウ設定（ウィンドウサイズを変更可能にする）
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1280, 720)

# ループ処理
target_fps = 30
frame_time = 1.0 / target_fps

try:
    while True:
        start_time = time.time()
        
        # 画像取得
        with ia.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            frame = np.array(component.data, dtype=np.uint8).reshape(component.height, component.width)
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)  # 必要に応じて変換
        
        # YOLOによる物体検出・トラッキング
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1, iou=0.45)
        
        # 検出結果の描画
        annotated_frame = frame.copy()
        if results is not None and len(results) > 0 and results[0].boxes is not None:
            if results[0].boxes.xyxy is not None:
                for box, track_id, cls in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [-1] * len(results[0].boxes.xyxy),
                    results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else [-1] * len(results[0].boxes.xyxy)
                ):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = f"ID: {int(track_id)} Class: {int(cls)}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("YOLOが何も検出できませんでした。")
        
        # 表示
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # FPS調整
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        # 終了キー
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("停止中...")
finally:
    ia.stop()
    ia.destroy()
    cv2.destroyAllWindows()
    print("終了しました。")
