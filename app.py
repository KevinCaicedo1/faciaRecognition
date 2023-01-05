from flask import Flask, jsonify, request,render_template,Response
import cv2
import numpy as np
import os
from openni import openni2

app = Flask(__name__) 
face_cascade = cv2.CascadeClassifier('C:/Users/Caicedo/Downloads/9+CodigoFuente/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


modelo='1850131556'
ruta1='C:/Users/Caicedo/Downloads/9+CodigoFuente/reconocimientofacial1/Data'


# # Start the infrared stream

  

def captureImage(identification):
    openni2.initialize()
    capture = openni2.Device.open_any()
    # Start the depth stream
    depth_stream = capture.create_depth_stream()
    rgb_stream = capture.create_color_stream()
    rutacompleta = ruta1 + '/'+ identification
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)
    depth_stream.start()
    rgb_stream.start()
    id=0
    while True:
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        # Get the frame from the color stream
        frame1 = rgb_stream.read_frame()
        frame_data1 = frame1.get_buffer_as_uint8()
        # Convert the frame data to an OpenCV image
        image1 = np.ndarray((frame1.height, frame1.width, 3), dtype=np.uint8, buffer=frame_data1)
        cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        imageCopy = gray.copy()   
        image = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
        image = np.uint8(image)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rostroCapturado = imageCopy[y:y+h,x:x+w]
            rostroCapturado = cv2.resize(rostroCapturado,(160,160),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(rutacompleta + '/imagen_{}.jpg'.format(id),rostroCapturado)
            id = id + 1
            x = int(x + w/2)
            y = int(y + h/2)
            # Mostrar el punto medio del rostro
            cv2.circle(image1, (x, y), 2, (0, 0, 255), 2)
            # Mostrar la distancia del rostro
            if(x>= 240):
                x = 239
            if(y>= 320):
                y = 319
            distance = image[x, y]
            # convertrir del rango de 0-255 a 100-3500
            distance = distance * (3500-100)/255 + 100
            # solo 2 decimales
            distance = round(distance, 2)
            # print(distance)
            cv2.putText(image1,str(distance/10)+" cm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        if(id==512):
            break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
    depth_stream.stop()
    rgb_stream.stop()
    openni2.unload()
    trainingModel()



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
    rutacompleta = ruta1 + '/'+ modelo
    return render_template('capture-photo.html')

def recognizeFace():
    openni2.initialize()
    entrenamientoEigenFaceRecognizer=cv2.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('modelEigenFace.xml')
    listaData=os.listdir(ruta1)
    capture = openni2.Device.open_any()
    # Start the depth stream
    depth_stream = capture.create_depth_stream()
    rgb_stream = capture.create_color_stream()
    depth_stream.start()
    rgb_stream.start()
    id=0
    while True:
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        # Get the frame from the color stream
        frame1 = rgb_stream.read_frame()
        frame_data1 = frame1.get_buffer_as_uint8()
        # Convert the frame data to an OpenCV image
        image1 = np.ndarray((frame1.height, frame1.width, 3), dtype=np.uint8, buffer=frame_data1)
        cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        imageCopy = gray.copy()

        image = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
        image = np.uint8(image)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rostroCapturado = imageCopy[y:y+h,x:x+w]
            rostroCapturado = cv2.resize(rostroCapturado,(160,160),interpolation=cv2.INTER_CUBIC)
            result = entrenamientoEigenFaceRecognizer.predict(rostroCapturado)
            cv2.putText(image1,'{}'.format(result), (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)
            if result[1] < 5700:
                cv2.putText(image1,'{}'.format(listaData[result[0]]), (x, y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
                cv2.putText(image1,'{}'.format(result), (x, y-5), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image1,'Desconocido', (x, y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)

            x = int(x + w/2)
            y = int(y + h/2)
            # Mostrar el punto medio del rostro
            cv2.circle(image1, (x, y), 2, (0, 0, 255), 2)
            # Mostrar la distancia del rostro
            if(x>= 240):
                x = 239
            if(y>= 320):
                y = 319
            distance = image[x, y]
            # convertrir del rango de 0-255 a 100-3500
            distance = distance * (3500-100)/255 + 100
            # solo 2 decimales
            distance = round(distance, 2)
            # print(distance)
            cv2.putText(image1,str(distance/10)+" cm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        if(id==512):
            break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

def trainingModel():
    dataPath = ruta1
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)
    faceSamples=[]
    ids = []
    id=0
    for imagePath in imagePaths:
        completePath = dataPath + '/' + imagePath
        print('completePath=',completePath)

        for file in os.listdir(completePath):
            print('file=',file)
            ids.append(id)
            faceSamples.append(cv2.imread(completePath+'/'+file,0))

        id = id + 1
    print('Iniciando el entrenamiento...espere')
    trainingEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    trainingEigenFaceRecognizer.train(faceSamples, np.array(ids))
    trainingEigenFaceRecognizer.write('modelEigenFace.xml')
    print('Modelo EigenFace entrenado')
    
