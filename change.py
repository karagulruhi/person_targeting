import cv2
import numpy as np
import onnxruntime as ort

model_path = "best_quantized.onnx" 
model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
stream_url = "http://192.168.2.49:4747/video"
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Görüntüyü model için hazırla
    img = cv2.resize(frame, (640, 640))
    input_tensor = img.transpose(2,0,1)[np.newaxis].astype(np.float32)/255.0
    
    # Tespit yap
    outputs = model.run(None, {"images": input_tensor})[0][0].T
    
    # Orijinal kutuları topla
    boxes = []
    for det in outputs:
        xc, yc, w, h, conf = det[:5]
        if conf > 0.5:  # İlk aşama için düşük güven eşiği
            # Orijinal görüntü boyutuna ölçeklendir
            x1 = int((xc - w/2) * frame.shape[1]/640)
            y1 = int((yc - h/2) * frame.shape[0]/640)
            x2 = int((xc + w/2) * frame.shape[1]/640)
            y2 = int((yc + h/2) * frame.shape[0]/640)
            boxes.append([x1, y1, x2, y2, conf])
    
    # NMS uygula
    if boxes:
        # OpenCV NMS için format dönüşümü
        nms_boxes = [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2, _) in boxes]
        scores = [conf for (_, _, _, _, conf) in boxes]
        
        # NMS parametreleri
        conf_threshold = 0.5
        nms_threshold = 0.4
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_threshold, nms_threshold)
        
        # Filtrelenmiş kutular
        final_boxes = [boxes[i] for i in indices.flatten()]
    else:
        final_boxes = []
    
    # Sonuçları çiz
    for box in final_boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow('Human Detection', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()