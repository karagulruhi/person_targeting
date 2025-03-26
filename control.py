import socket
import pickle
import os
from adafruit_servokit import ServoKit
import time
# Unix domain socket dosyasını sil (varsa)
SOCKET_PATH = '/tmp/robot_socket'
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# Servo motor kontrolü için kurulum
kit = ServoKit(channels=16)
kit.servo[0].angle = 90  # Başlangıç pozisyonu
kit.servo[1].angle = 90 
kit.servo[2].angle = 90
   
area_low=24722
area_h=227965


current_servo0_angle = 90
current_servo1_angle = 90 
current_servo2_angle = 90
  # Başlangıç açısı (ortada başlat)
current_x = None  # Son x değerini takip etmek için
MIN_MOVEMENT_THRESHOLD = 5  # X'deki minimum değişim miktarı
def olceklendir(deger, min_giris, max_giris, min_cikis, max_cikis):
    return max(min_cikis, min(max_cikis, 
        (deger - min_giris) * (max_cikis - min_cikis) / (max_giris - min_giris) + min_cikis
    ))
def servo_kontrol_et(x, y,area):
    global current_servo0_angle,current_servo1_angle, current_servo2_angle, current_x
    x=480-x
    # İlk çağrıda current_x None olacak, bu durumu kontrol et
    if current_x is None:
        current_x = x
        return
    print(area)
    print(y)
    try:
        if x < 240 and area < 126344:
            servo0_aci = olceklendir(240 - x, 0, 480, 0, 30) 
            servo1_aci = olceklendir(area, 24722, 227965, 0, 30)
            servo2_aci = olceklendir(y, 0, 640, 0, 30) 
            current_servo0_angle = min(180, current_servo0_angle + servo0_aci)
            current_servo1_angle = max(60, current_servo1_angle - servo1_aci)
            current_servo2_angle = min(120, current_servo1_angle + servo1_aci)
            print(f"Şahıs uzakta {servo1_aci}°istenilen açı {current_servo1_angle}, {current_servo1_angle}")

        elif x > 240 and area < 126344:
            servo0_aci = olceklendir(x - 240, 0, 480, 0, 30)
            servo1_aci = olceklendir(area, 24722, 227965, 0, 30)
            servo2_aci = olceklendir(y, 0, 640, 0, 30) 
            current_servo0_angle = max(0, current_servo0_angle - servo0_aci)
            current_servo1_angle = max(60, current_servo1_angle -servo1_aci)
            current_servo2_angle = min(120, current_servo1_angle + servo1_aci)
            print(f"Şahıs uzakta {servo1_aci}°istenilen açı {current_servo1_angle} {current_servo1_angle}")

        elif x < 240 and area >= 126344:
            servo0_aci = olceklendir(240 - x, 0, 480, 0, 30) 
            servo1_aci = olceklendir(area, 24722, 227965, 0, 30)
            servo2_aci = olceklendir(y, 0, 640, 0, 30) 
            current_servo0_angle = min(180, current_servo0_angle + servo0_aci)
            current_servo1_angle = min(120, current_servo1_angle + servo1_aci)
            current_servo2_angle = max(60, current_servo1_angle -servo1_aci)
            print(f"(sahıs yakında {servo1_aci}°istenilen açı {current_servo1_angle}")

        elif x > 240 and area >= 126344:
            servo0_aci = olceklendir(x - 240, 0, 480, 0, 30)
            servo1_aci = olceklendir(area, 24722, 227965, 0, 30)
            servo2_aci = olceklendir(y, 0, 640, 0, 30) 
            current_servo0_angle = max(0, current_servo0_angle - servo0_aci)
            current_servo1_angle = min(120, current_servo1_angle + servo1_aci)
            current_servo2_angle = max(60, current_servo1_angle - servo1_aci)
            print(f"(sahıs yakında {servo1_aci}°istenilen açı {current_servo1_angle}")

        kit.servo[0].angle = current_servo0_angle
        time.sleep(0.1)
        kit.servo[1].angle = current_servo1_angle
        time.sleep(0.1)  # Daha kısa bekleme süresi
        kit.servo[2].angle = current_servo2_angle
        time.sleep(0.1)  # Daha kısa bekleme süresi
        
    
    finally:
        current_x = x  # X değerini her durumda güncelle

# Unix domain socket oluştur
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(SOCKET_PATH)
sock.listen(3)
print("Socket dinleniyor...")

# Sonsuz döngü içinde bağlantıları kabul et
while True:
    conn, _ = sock.accept()
    print("Yeni bağlantı kabul edildi.")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("Bağlantı kesildi. Yeniden dinleniyor...")
                break

            try:
                data = pickle.loads(data)
                if data.get('x_center', 0)!=None:
                    x = data.get('x_center', 0)
                    y = data.get('y_center', 0)
                    area=  data.get('area', 0)
                    servo_kontrol_et(x, y,area)
            except pickle.UnpicklingError:
                print("Geçersiz veri alındı.")
            except Exception as e:
                print(f"Beklenmeyen hata: {e}")
    finally:
        if conn:
            conn.close()

# Çıkışta temizleme

sock.close()
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)
print("Socket kapatıldı ve temizlendi.")
