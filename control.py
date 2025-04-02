import socket
import pickle
import os
import time
from adafruit_servokit import ServoKit

# Unix domain socket dosyasını temizle
SOCKET_PATH = '/tmp/robot_socket'
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# Servo motor kurulum
kit = ServoKit(channels=16)
def set_servo_positions():
    for i in range(3):
        kit.servo[i].angle = 90  # Başlangıç açısı
set_servo_positions()

# Ölçeklendirme fonksiyonu
def olceklendir(deger, min_giris, max_giris, min_cikis, max_cikis):
    return max(min_cikis, min(max_cikis, (deger - min_giris) * (max_cikis - min_cikis) / (max_giris - min_giris) + min_cikis))

# Servo motor açılarını kontrol eden fonksiyon
current_servo_angles = [90, 90, 90]
def servo_kontrol_et(x, y, area):
    global current_servo_angles
    x = 480 - x
    y= 640 - y

    servo0_aci = olceklendir(abs(240 - x), 0, 480, 0, 30)
    servo1_aci = olceklendir(abs(126344-area),24722, 227965, 0, 30)
    servo2_aci = olceklendir(abs(320-y), 0, 640, 0, 30)
    
    
    if area < 126344:
        current_servo_angles[0] = min(180, current_servo_angles[0] + servo0_aci) if x < 240 else max(0, current_servo_angles[0] - servo0_aci)
        current_servo_angles[1] = max(60, current_servo_angles[1] - servo1_aci)
        current_servo_angles[2] = max(60, current_servo_angles[2] -5)
    else:
        current_servo_angles[0] = min(180, current_servo_angles[0] + servo0_aci) if x < 240 else max(0, current_servo_angles[0] - servo0_aci)
        current_servo_angles[1] = min(120, current_servo_angles[1] + servo1_aci)
    
    current_servo_angles[2] = min(120, current_servo_angles[2] +servo2_aci) if y < 320 else max(60, current_servo_angles[2] - servo2_aci)
    
    for i in range(3):
        kit.servo[i].angle = current_servo_angles[i]
        time.sleep(0.05)

# Unix domain socket oluştur
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(SOCKET_PATH)
sock.listen(5)
print("Socket dinleniyor...")

while True:
    conn, _ = sock.accept()
    print("Yeni bağlantı kabul edildi.")
    
    while True:
        data = conn.recv(1024)
        if not data:
            print("Bağlantı kesildi. Yeniden dinleniyor...")
            break
        
        try:
            data = pickle.loads(data)
            
            if 'x_center' in data:
                servo_kontrol_et(data['x_center'], data['y_center'], data['area'])
                
        except pickle.UnpicklingError:
            print("Geçersiz veri alındı.")
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
    
    conn.close()

# Çıkışta temizleme
sock.close()
os.remove(SOCKET_PATH)
print("Socket kapatıldı ve temizlendi.")
