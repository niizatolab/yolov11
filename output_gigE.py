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

# ウィンドウの設定（両方のウィンドウを画素数5328×3040に調整）
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 5328, 3040)
cv2.namedWindow("Processed Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processed Output", 5328, 3040)

# 処理ループ（目標FPS:32）
target_fps = 32
frame_time = 1.0 / target_fps

try:
    while True:
        start_time = time.time()
        
        # カメラから画像取得
        with ia.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            frame = np.array(component.data, dtype=np.uint8).reshape(component.height, component.width)
            # BayerパターンからBGR変換（必要に応じて変更してください）
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        
        # YOLOによる物体検出・トラッキング（bytetrack等のトラッカーを利用）
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1, iou=0.45)
        
        # 黒色の背景画像を作成（元フレームと同じサイズ）
        processed_img = np.zeros_like(frame)
        
        # 検出結果がある場合、元フレームにバウンディングボックスとラベル、また背景画像に赤の領域を描画
        if results[0].boxes is not None:
            # バウンディングボックス情報（xyxy形式）
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            # トラッキングID（存在しない場合は-1）
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [-1] * len(bboxes)
            
            for box, track_id in zip(bboxes, ids):
                x1, y1, x2, y2 = map(int, box[:4])
                # 元フレームに四角形の描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 中心座標の計算
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # ラベルにIDと中心座標を表示（フォントサイズや太さは必要に応じて調整）
                label = f"ID: {int(track_id)} Center: ({center_x}, {center_y})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 黒色背景の画像上で、バウンディングボックス内領域を赤色でハイライト
                # ※対象領域が画像のサイズ内に収まるか確認する必要があります
                processed_img[y1:y2, x1:x2] = (0, 0, 255)
                
        # それぞれのウィンドウに画像を表示
        cv2.imshow("YOLO Detection", frame)
        cv2.imshow("Processed Output", processed_img)
        
        # FPS調整（実行にかかった時間に応じ、次のフレーム更新までスリープ）
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        # 'q'キーでループ終了
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("停止中...")
finally:
    ia.stop()
    ia.destroy()
    cv2.destroyAllWindows()
    print("終了しました。")
