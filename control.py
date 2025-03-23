import socket
import pickle
import os
from adafruit_servokit import ServoKit

# Socket dosyasını sil (varsa)
if os.path.exists('/tmp/robot_socket'):
    os.remove('/tmp/robot_socket')

# Servo motor kontrolü için kurulum
kit = ServoKit(channels=8)  # FeatherWing için 8 kanal
kit.servo[1].angle = 90  # Başlangıç pozisyonu

# Unix domain socket oluştur ve dinle
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    sock.bind('/tmp/robot_socket')
    sock.listen(1)
    print("Socket dinleniyor...")
    conn, addr = sock.accept()
    print("Bağlantı kabul edildi.")

    while True:
        # Veriyi al
        data = conn.recv(1024)
        if not data:
            print("Boş veri alındı. Bağlantı kesildi.")
            break

        try:
            # Veriyi işle
            data = pickle.loads(data)
            print("Alınan Veri:", data)

            # Gelen veriye göre servo motoru kontrol et
            area = data.get('area', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)

            # Örnek: Alan büyüdükçe servo açısını artır
            if area > 5000:
                kit.servo[1].angle = 180  # Maksimum açı
            elif area > 2000:
                kit.servo[1].angle = 90  # Orta açı
            else:
                kit.servo[1].angle = 0  # Minimum açı

        except pickle.UnpicklingError:
            print("Geçersiz veri alındı.")
            continue
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
            continue

finally:
    # Soketi kapat ve dosyayı sil
    conn.close()
    sock.close()
    if os.path.exists('/tmp/robot_socket'):
        os.remove('/tmp/robot_socket')
    print("Socket kapatıldı ve temizlendi.")
