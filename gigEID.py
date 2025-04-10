import cv2
import numpy as np
import time
from harvesters.core import Harvester
from ultralytics import YOLO

# GigEカメラのセットアップ
def setup_gige_camera():
    h = Harvester()
    # Allied Visionのカメラ用CTIファイル（環境に合わせてパスを変更してください）
    h.add_file("C:\\Program Files\\Allied Vision\\Vimba_6.0\\VimbaGigETL\\Bin\\Win64\\VimbaGigETL.cti")
    h.update()
    ia = h.create_image_acquirer(0)
    ia.start()
    return ia

# YOLOv11モデルのロード（適切なパスに変更してください）
model = YOLO("yolo11n.pt")

# カメラを初期化
ia = setup_gige_camera()
print("GigEカメラが接続されました。")

# 画角調整：ウィンドウサイズを指定（必要に応じて変更）
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 5328, 3040)

# ループ処理（対象のFPS:32）
target_fps = 32
frame_time = 1.0 / target_fps

try:
    while True:
        start_time = time.time()
        
        # 画像取得
        with ia.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            frame = np.array(component.data, dtype=np.uint8).reshape(component.height, component.width)
            # BayerパターンからBGR変換（必要に応じて変更してください）
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        
        # YOLOによる物体検出・トラッキング
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1, iou=0.45)
        
        # 検出結果がある場合、バウンディングボックスおよびIDの描画
        if results[0].boxes is not None:
            # バウンディングボックス（xyxy形式）
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            # トラッキングID（存在すれば取得、なければ-1を設定）
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [-1] * len(bboxes)
            for box, track_id in zip(bboxes, ids):
                x1, y1, x2, y2 = map(int, box[:4])
                # 四角形を描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # トラッキングIDを描画
                label = f"ID: {int(track_id)}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 表示
        cv2.imshow("YOLO Detection", frame)
        
        # FPS調整（次のフレーム取得までの待機時間の調整）
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        # 'q'キーで終了
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("停止中...")
finally:
    ia.stop()
    ia.destroy()
    cv2.destroyAllWindows()
    print("終了しました。")
