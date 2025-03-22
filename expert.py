import cv2
import numpy as np
import onnxruntime as ort
import time




# ONNX modelini yükle
model_path = "best.onnx" 
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # GPU için "CUDAExecutionProvider"
stream_url = "http://192.168.2.49:4747/video"
cap = cv2.VideoCapture(stream_url) # 0 varsayılan kamera, IP kamera için URL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Performans ölçümü için
fps_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ön işleme
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    input_tensor = img_resized / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    
    # Model çıkarımı
    outputs = session.run(None, {"images": input_tensor})
    predictions = np.squeeze(outputs[0]).T  # (8400, 5)

    # Tespitleri filtrele
    def filter_predictions(predictions, confidence_threshold=0.25):
        scores = predictions[:, 4]
        mask = scores > confidence_threshold
        boxes_xywh = predictions[mask, :4]
        scores = scores[mask]
        
        if boxes_xywh.size == 0:
            return np.empty((0, 4)), np.empty((0,))
        
        # XYWH -> XYXY
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2]/2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3]/2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]/2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]/2
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 0.25, 0.45)
        return (boxes_xyxy[indices], scores[indices]) if indices.size > 0 else (np.empty((0, 4)), np.empty((0,)))

    boxes, scores = filter_predictions(predictions)
    
    # Bounding box çiz
    h, w = frame.shape[:2]
    scale_x = w / 640
    scale_y = h / 640
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        print(x1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person: {score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # FPS Hesapla
    fps_counter += 1
    if (time.time() - start_time) > 1:
        fps = fps_counter / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fps_counter = 0
        start_time = time.time()
    
    # Görüntüyü göster
    cv2.imshow('Real-Time Human Detection', frame)
    
    # 'q' ile çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()