import cv2  # pip install opencv-python
from cvzone.FaceDetectionModule import FaceDetector  # pip install cvzone

video = cv2.VideoCapture(1)  # Iniciamos a captura de vídeo
detector = FaceDetector(minDetectionCon=0.5)  # Iniciamos o detector de faces

while True:
    _, frame = video.read()  # Capturamos o frame
    frame, boxes = detector.findFaces(frame, draw=False)  # Detectamos as faces
    img = frame.copy()  # Copiamos o frame para não alterar o original
    if boxes:  # Se houverem faces detectadas
        for box in boxes:  # Loop pelas faces detectadas
            x, y, w, h = box['bbox']  # Pegamos as coordenadas da face
            rec = frame[y:y+h, x:x+w]  # Recortamos a face
            recBlur = cv2.blur(rec,  (30,30))  # Aplicamos o blur
            img[y:y+h,x:x+w] = recBlur  # Colocamos a face de volta na imagem

    cv2.imshow('Video', img)  # Mostramos a imagem
    cv2.waitKey(1)  # Esperamos 1ms