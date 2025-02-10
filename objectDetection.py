import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import torch
import cv2
import io

import pathlib
import sys
from pathlib import Path

# Corrige o problema de WindowsPath no macOS
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath


from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()


# Diretório atual do arquivo objectDetection.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho da pasta yolov5 dentro do projeto
yolov5_dir = os.path.join(current_dir, "yolov5")

# Adiciona o diretório raiz do projeto e o yolov5 ao sys.path
sys.path.append(current_dir)
sys.path.append(yolov5_dir)


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_segments
from yolov5.utils.torch_utils import select_device

model_path = "yolov5/runs/train/twenty-epochs/weights/best.pt"

EMAIL_SENDER = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"]
SMTP_SERVER = os.environ["SMTP_SERVER"]
SMTP_PORT =os.environ["SMTP_PORT"]

def send_email_alert(object_detected, image):
    try:
        subject = "[Alerta] Possível ameaça!"
        body = f"Objeto detectado: {object_detected}"

        message = MIMEMultipart()
        message["From"] = EMAIL_SENDER
        message["To"] = EMAIL_RECEIVER
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEImage(image,  name="alerta.jpg"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
            print("Alerta enviado com sucesso!")

    except Exception as e:
        print(f"Erro ao enviar o alerta: {e}")

def detect_objects(input_path, model_path):
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device)
    dataset = LoadImages(input_path)
    counter = 0
    object_type = '' 

    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img)
        img = img.float() / 255.0
        img = img.to(device)

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        predictions = model(img) 
        predictions = non_max_suppression(predictions)

        for pred in predictions:
            if pred is not None and len(pred):
                pred[:, :4] = scale_segments(img.shape[2:], pred[:, :4], im0s.shape).round()

                success, buffer = cv2.imencode(".jpg", im0s)
                if not success:
                    print("Falha ao processar imagem")
                    continue

                image_bytes = buffer.tobytes()

                for *box, conf, cls in pred:
                    label = model.names[int(cls)]
                
                    # if object_type != label:
                    #     counter = 0

                    object_type = label

                    if conf >= 0.5:   
                        print(f"Objeto detectado: {label} com assertividade {conf:.2f}")
                        if (object_type == "objeto cortante" or object_type == "arma de fogo") and counter == 0:
                            counter = counter + 1
                            send_email_alert(object_type, image_bytes)
                    else:
                        # object_type = "objeto anomalo"
                        counter = 0
                        print(f"Objeto anômalo com assertividade {conf:.2f}")     

def main():
    print("Iniciando sistema de detecção de objetos...")
    video_test_path = "videos_teste/video.mp4"
    detect_objects(video_test_path, model_path)

if __name__ == "__main__":
    main()