import cv2
import numpy as np
import time
from harvesters.core import Harvester
from ultralytics import YOLO


#yes
# GigEカメラのセットアップ
def setup_gige_camera():
    h = Harvester()
    h.add_file("C:\Program Files\Allied Vision\Vimba_6.0\VimbaGigETL\Bin\Win64\VimbaGigETL.cti")  # 適切なパスに変更
    h.update()
    ia = h.create_image_acquirer(0)
    ia.start()
    return ia

# YOLOv11モデルのロード
model = YOLO("yolo11n.pt")  # 適切なパスに変更

# カメラを初期化
ia = setup_gige_camera()
print("GigEカメラが接続されました。")

#画角調整
cv2.namedWindow("YOLO Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection",5328,3040)

# ループ処理
target_fps = 32
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
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",conf=0.1, iou=0.45)
        #results = model(image)
        
        # 検出結果の描画
        if results[0].boxes is not None:
           for box in results[0].boxes.xyxy.cpu().numpy():
               x1, y1, x2, y2 = map(int, box[:4])
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # annotated_frame = frame.copy()
        # if results is not None and len(results) > 0 and results[0].boxes is not None:
        #     if results[0].boxes.xyxy is not None:
        #         for box, track_id, cls in zip(
        #             results[0].boxes.xyxy.cpu().numpy(), 
        #             results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [-1] * len(results[0].boxes.xyxy), 
        #             results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else [-1] * len(results[0].boxes.xyxy)
        #         ):
        #             x1, y1, x2, y2 = map(int, box[:4])
        #             label = f"ID: {int(track_id)} Class: {int(cls)}"
        #             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # else:
        #     print("検出なし")
        # 表示
        cv2.imshow("YOLO Detection", frame)

        
        # FPS調整
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        # 終了キー
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("停止中...")
finally:
    ia.stop()
    ia.destroy()
    cv2.destroyAllWindows()
    print("終了しました。")