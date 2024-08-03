from flask import Flask, render_template, Response
import cv2
import RPi.GPIO as GPIO
import time
import threading
import requests
import pyttsx3

app = Flask(__name__)

# Configuración de los pines GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

TRIG = 23
ECHO = 24
BUZZER = 18
PROXIMITY_LED = 25
OBJECT_DETECTED_LED = 17

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(PROXIMITY_LED, GPIO.OUT)
GPIO.setup(OBJECT_DETECTED_LED, GPIO.OUT)

# Configuración de la cámara y el modelo de detección de objetos
classNames = []
classFile = "/home/hermosillopi/Documents/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/hermosillopi/Documents/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/hermosillopi/Documents/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(160, 160)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

engine = pyttsx3.init()

def getObjects(img, thres, nms, draw=True, objects=[]):

    # Detecta objetos en la imagen con una umbral de confianza (thres) y una umbral de supresión de no máximos (nms)
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # Si no se proporcionan objetos específicos para detectar, usa todas las clases disponibles
    if len(objects) == 0: objects = classNames
    objectInfo = []  # Lista para almacenar los objetos detectados
    detected_objects = set()  # Set para almacenar objetos únicos detectados
    maxConfidence = 0  # Almacena la confianza máxima detectada
    maxConfidenceObject = None  # Almacena la información del objeto con la máxima confianza detectada
    # Verifica e itera sobre los objetos detectados
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            # Verifica si el objeto está en la lista de objetos a detectar y no ha sido detectado antes
            if className in objects and className not in detected_objects:
                detected_objects.add(className)  # Agregar el objeto al set de objetos detectados
                # Actualiza el objeto con máxima confianza
                if confidence > maxConfidence:
                    maxConfidence = confidence
                    maxConfidenceObject = (box, className, confidence)
                # Dibuja rectángulos y texto sobre la imagen para los objetos detectados
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # Agrega el objeto con la máxima confianza a la lista y convierte su nombre a voz
        if maxConfidenceObject:
            box, className, confidence = maxConfidenceObject
            objectInfo.append([box, className, confidence])
            print(f"Detected object: {className}, Confidence: {round(confidence * 100, 2)}%")
            engine.say(className)  # Convertir el nombre del objeto a voz
            engine.runAndWait()
    # Devuelve la imagen con los dibujos y la información de los objetos detectados
    return img, objectInfo

def distance():
    GPIO.output(TRIG, False)
    time.sleep(2)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    dist = pulse_duration * 17150
    dist = round(dist, 2)

    return dist

# Función para leer datos del GPS y enviarlos al servidor
def enviar_gps_al_servidor():
    while True:

        # Enviar datos al servidor
        data = {'latitude': latitude, 'longitude': longitude}
        try:
            response = requests.post('http://192.168.1.160:5001/gps', json=data)
            if response.status_code == 200:
                print("Datos GPS enviados con éxito")
            else:
                print("Error al enviar los datos GPS:", response.status_code)
        except Exception as e:
            print("Error al conectar con el servidor:", e)

        time.sleep(10)  # Enviar datos cada 10 segundos

# Crear un hilo para enviar los datos GPS al servidor
gps_thread = threading.Thread(target=enviar_gps_al_servidor)
gps_thread.daemon = True
gps_thread.start()

# Generador de frames de video
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            success, img = cap.read()

            # Detección de objetos
            if not success:
                break
            result, objectInfo = getObjects(img, 0.45, 0.2, objects=['person', 'car', 'stop sign', 'traffic light'])

            # Led de objetos detectados por la cámara
            if objectInfo:
                GPIO.output(OBJECT_DETECTED_LED, True)
            else:
                GPIO.output(OBJECT_DETECTED_LED, False)

            # Configuración del sensor de proximidad
            dist = distance()
            print("Distance:", dist, "cm")

            if dist < 15:
                GPIO.output(BUZZER, True)
                GPIO.output(PROXIMITY_LED, True)
                print("Possible collision")
            else:
                GPIO.output(BUZZER, False)
                GPIO.output(PROXIMITY_LED, False)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                break

    except KeyboardInterrupt:
        print("Measurement stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)