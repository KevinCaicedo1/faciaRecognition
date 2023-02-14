import shutil
from flask import Flask, jsonify, request, render_template, Response
import cv2
import numpy as np
# import tensorflow as tf
import os
from openni import openni2
from sklearn.model_selection import train_test_split
import pathlib
import face_recognition
from flask_socketio import SocketIO, emit
# add cors 
from flask_cors import CORS
import eventlet
import eventlet.wsgi
import socketio
import base64


sio = socketio.Server(cors_allowed_origins="*")
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
# add cors para todos los dominios
dict1 = {}
i=0
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')


modelo = '1850131556'
ruta1 = 'C:/Users/Caicedo/Downloads/9+CodigoFuente/reconocimientofacial1/Data'
listaData=os.listdir(ruta1+'/train')


@socketio.on('stream-request')
def handle_stream_request(frame):
    copyFrame = frame
    frame = frame[23:]
    frame = base64.b64decode(frame)
    # convert string of image data to uint8
    nparr = np.fromstring(frame, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # roi_gray = gray[y:y+h, x:x+w]
        #     # reconocimiento
        #     rostrocapturado = cv2.resize(roi_gray, (224,224),
        #                                     interpolation=cv2.INTER_CUBIC)
        #     resultado = entrenamientoEigenFaceRecognizer.predict(rostrocapturado)

        #     if resultado[1] < 8000:
        #         cv2.putText(img, '{}'.format(listaData[resultado[0]]), (x, y-20), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
        #     else:
        #         cv2.putText(img, 'Desconocido', (x, y-20), 2, 1.1, (0, 0, 255), 1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', img)
        frame = base64.b64encode(buffer)
        frame = 'data:image/jpeg;base64,' + frame.decode('utf-8')
        socketio.emit('stream-response', frame)
    else:
        socketio.emit('stream-response', copyFrame)



face_cascade = cv2.CascadeClassifier(
    'C:/Users/Caicedo/Downloads/9+CodigoFuente/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()


batch_size = 32
img_height = 224
img_width = 224

pathDir = pathlib.Path(ruta1+'/train')
classnames = np.array(
    [item.name for item in pathDir.glob('*') if item.name != "LICENSE.txt"])
print('classnames=', classnames)



def loadImages(path):
    # return array of images
    datasetDir = os.listdir(path)
    x = []
    y = []
    i = 0
    labels = dict()
    for folder in datasetDir:
        labels[i] = folder
        images = os.listdir(path + '/' + folder)
        for image in images:
            img = cv2.imread(path + '/' + folder + '/' + image)
            img = img.astype('float32')/255
            img = cv2.resize(img, (img_height, img_width),
                             interpolation=cv2.INTER_AREA)
            x.append(img)
            y.append(i)
        i = i + 1

    return x, y, labels


modelLabels = dict()



def captureImage(identification):

    rutacompleta = ruta1 + '/train/' + identification
    rutaTest = ruta1 + '/val/' + identification
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)
    if not os.path.exists(rutaTest):
        os.makedirs(rutaTest)
    video = cv2.VideoCapture('http://192.168.50.62:4747/video')

    # Check if camera opened successfully
    if (video.isOpened() == False):
        print("Unable to read camera feed")
        return

    id = 0
    while True:
        ret, frame = video.read()
        image1 = np.ndarray(
            (frame.shape[0], frame.shape[1], 3), dtype=np.uint8, buffer=frame)
        cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        imageCopy = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(imageCopy, 1.3, 5)
        for (x, y, w, h) in faces:

            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rostroCapturado = imageCopy[y:y+h, x:x+w]
            # ndim 4 para que funcione con el modelo
            if (id > 60):
                rostroCapturado = cv2.resize(
                    rostroCapturado, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(rutaTest + '/imagen_{}.jpg'.format(id),
                            rostroCapturado)
            else:
                rostroCapturado = cv2.resize(
                    rostroCapturado, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(rutacompleta + '/imagen_{}.jpg'.format(id),
                            rostroCapturado)

            id = id + 1
            x = int(x + w/2)
            y = int(y + h/2)
            if (x >= 240):
                x = 239
            if (y >= 320):
                y = 319

        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        # tomar 320 fotos y salir
        if id >= 80:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
    # verificamos si el modelo existe
    if os.path.isfile('model.xml'):
        # actualizamos el modelo
        trainingModel()
    else:
        # si no existe creamos el modelo
        trainingModel()
        print("Modelo cargado")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture-photo', methods=['GET'])
def capturePhoto():
    identification = request.args.get('id')
    return Response(captureImage(identification), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognize')
def recognizeVideo():
    return Response(recognizeFace(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/formperson')
def formperson():
    return render_template('formperson.html')


@app.route('/formperson', methods=['POST'])
def formperson_post():
    name = request.form['name']
    lastname = request.form['lastname']
    age = request.form['age']
    return render_template('formperson.html', name=name, lastname=lastname, age=age)


@app.route('/capture-photo')
def capture_photo():
    rutacompleta = ruta1 + '/' + modelo
    return render_template('capture-photo.html')


def recognizeFace():
    video = cv2.VideoCapture('http://192.168.50.62:4747/video')
    id = 0
    listaData=os.listdir(ruta1+'/train')
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Convert the frame data to an OpenCV image
        image1 = frame
        image1 = np.ndarray(
            (frame.shape[0], frame.shape[1], 3), dtype=np.uint8, buffer=frame)
        imageCopy = image1.copy()
        imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
            # face = np.reshape(face, (1,50176))
            preds = entrenamientoEigenFaceRecognizer.predict(face)
            print(preds)
            cv2.putText(image1, '{}'.format(preds), (x, y-5),
                        1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)

            if preds[1] < 8000:
                cv2.putText(image1, '{}'.format(
                listaData[preds[0]]), (x, y-20), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image1, "No encontrado", (x, y-20),
                       2, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            x = x + w//2
            y = y + h//2
            if (x >= 240):
                x = 239
            if (y >= 320):
                y = 319
        

        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        if (id == 512):
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


def trainingModel():
    dataPath = ruta1 + '/train'
    testPath = ruta1 + '/val'
    imagePaths = os.listdir(dataPath)
    print('imagePaths=', imagePaths)
    faceSamples = []
    ids = []
    id = 0
    for imagePath in imagePaths:
        completePath = dataPath + '/' + imagePath
        print('completePath=', completePath)

        for file in os.listdir(completePath):
            print('file=', file)
            ids.append(id)
            faceSamples.append(cv2.imread(completePath + '/' + file, 0))

        id = id + 1

    ids = np.array(ids)
    print('Entrenando el modelo EigenFace...')
    print('Iniciando el entrenamiento...espere')
    trainingEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    trainingEigenFaceRecognizer.train(faceSamples, ids)
    trainingEigenFaceRecognizer.write('model.xml')
    print('Modelo EigenFace entrenado')
    entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('model.xml')
    
# cargar el modelo si existe
if os.path.isfile('model.xml'):
    print('Existe el modelo, se cargara')
    entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('model.xml')
else:
    print('No existe el modelo, se creara uno nuevo')
    trainingModel()



def updateXMLModel():
    dataPath = ruta1 + '/train'
    imagePaths = os.listdir(dataPath)
    print('imagePaths=', imagePaths)
    faceSamples = []
    ids = []
    id = 0
    for imagePath in imagePaths:
        completePath = dataPath + '/' + imagePath
        print('completePath=', completePath)

        for file in os.listdir(completePath):
            print('file=', file)
            ids.append(id)
            faceSamples.append(cv2.imread(completePath+'/'+file, 0))

        id = id + 1
    print('Iniciando la actualizacion del modelo...espere')
    entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('model.xml')
    entrenamientoEigenFaceRecognizer.update(faceSamples, np.array(ids))
    entrenamientoEigenFaceRecognizer.write('model.xml')
    print('Modelo EigenFace actualizado')


socketio.run(app, host='localhost', port=5000, debug=True)