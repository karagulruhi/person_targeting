import cv2
import numpy as np

import socket
import pickle

import tflite_runtime.interpreter as tflite

# TFLite modelini yükle
interpreter = tflite.Interpreter(model_path="yolo8n/yolo8n128_float16.tflite")




interpreter.allocate_tensors()

# Giriş ve çıkış tensörlerini al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
person_detected = False  #
# Kamera akış URL'si
stream_url = "http://192.168.1.2:4747/video"
frame_skip = 8# Her 8 karede bir işle
frame_count = 0
# Giriş boyutunu öğren
input_shape = input_details[0]['shape']
img_size = (input_shape[1], input_shape[2])

PERSON_CLASS_ID = 0
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/robot_socket')
# Kamera aç
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Genişlik 320 piksel
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Yükseklik 240 piksel

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, img_size)
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0

    # Modeli çalıştır
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    person_detected_in_frame = False
    output_data = interpreter.get_tensor(output_details[0]['index'])


    # Çıktıyı işle (örneğin, sınırlayıcı kutuları çiz)
    for det in output_data[0]:
        
        x1, y1, x2, y2, conf,id= det[0], det[1], det[2], det[3], det[4],det[5]
        if  conf > 0.6:  # Sadece insan sınıfı ve güvenilirlik eşiği
          
            # Sınırlayıcı kutuyu orijinal görüntü boyutuna ölçeklendir
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
     
    

            area = (x2-x1)*(y2-y1)
            try:
              data = {'area': area, 'x_center': center_y, 'y_center': center_x}
              sock.sendall(pickle.dumps(data))
            except Exception as e:
                print(f"Soket gönderme hatası: {e}")
    if not person_detected_in_frame:
        if person_detected:  # Önceki karede tespit edilmişse
            try:
                # İnsan olmadığını belirten veri gönder
                data = {'area': None, 'x_center': None, 'y_center': None}
                sock.sendall(pickle.dumps(data))
                 # current_x'i sıfırla
                print("İnsan tespit edilmedi")
            except Exception as e:
                print(f"Soket gönderme hatası: {e}")
        
        person_detected = False
    else:
        person_detected = True
# Soketi kapat)
            

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)



    cv2.imshow("Human Detection", frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
sock.close()
cap.release()
cv2.destroyAllWindows()
