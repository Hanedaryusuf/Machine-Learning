import cv2
import numpy as np
from deepface import DeepFace
from twilio.rest import Client
import os
import time

# Twilio API bilgileri
account_sid = "Your_SID_Code"  
auth_token = "Your_Auth_Toke"    
twilio_phone_number = "+1 234 567 89" 
to_phone_number = "+9123456789"   

# Twilio istemcisi
client = Client(account_sid, auth_token)

def send_sms():
    try:
        message = client.messages.create(
            body="Eşleşen yüz algılandı!",
            from_=twilio_phone_number,
            to=to_phone_number
        )
        print(f"SMS gönderildi: {message.sid}")
    except Exception as e:
        print(f"SMS gönderilemedi: {e}")

def extract_face_features(reference_images_path):
    """
    Referans görüntülerinin yüz özelliklerini çıkarır ve bellekte saklar.
    """
    reference_features = []
    reference_images = [os.path.join(reference_images_path, f) for f in os.listdir(reference_images_path) if f.endswith(('.jpg', '.png'))]

    if not reference_images:
        print("Referans görüntüler bulunamadı!")
        return None

    print("Referans yüz özellikleri çıkarılıyor...")
    for image_path in reference_images:
        try:
            features = DeepFace.represent(img_path=image_path, model_name='VGG-Face', detector_backend='opencv')
            reference_features.append(features)
        except Exception as e:
            print(f"Özellik çıkarma hatası: {e}")
    
    return reference_features

def start_custom_stream(reference_images_path, reference_features):
    """
    Kameradan alınan kareleri referans yüzlerle karşılaştırarak eşleşme kontrolü yapar.
    """
    cap = cv2.VideoCapture(0)  # Kamerayı başlat
    matched = False  # Eşleşme bayrağı

    print("Yüz eşleştirme başlatıldı...")

    while cap.isOpened() and not matched:
        ret, frame = cap.read()
        if not ret:
            break

        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        try:
            # Yüz özelliklerini çıkar
            frame_features = DeepFace.represent(img_path=temp_image_path, model_name='VGG-Face', detector_backend='opencv')
        except Exception as e:
            print(f"Yüz çıkarımı hatası: {e}")
            continue

        # Referans yüzlerle karşılaştırma
        for ref_features in reference_features:
            # Burada Cosine mesafesi ile karşılaştırma yapıyoruz (veya farklı bir mesafe metodu kullanabilirsiniz)
            distance = DeepFace.distance(frame_features, ref_features, metric='cosine')
            if distance < 0.4:  # Eşik değeri
                print(f"Eşleşme bulundu!")
                send_sms()  # SMS gönder
                matched = True
                break

        cv2.imshow("Kamera", frame)

        # 'q' tuşuna basarak döngüyü manuel durdurabilirsiniz
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    if matched:
        print("Eşleşme bulundu, akış durduruldu.")
    else:
        print("Eşleşme bulunamadı.")

# Referans görüntülerin bulunduğu klasör
reference_images_path = "C:/test"
reference_features = extract_face_features(reference_images_path)

if reference_features:
    start_custom_stream(reference_images_path, reference_features)
