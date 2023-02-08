from flask import Flask, jsonify, request, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import os
from openni import openni2
from sklearn.model_selection import train_test_split
import pathlib
import face_recognition
import keras
# SENet50 model
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.resnet import decode_predictions
# from keras.applications.resnet import ResNet50
from keras.utils.np_utils import to_categorical
# SGD
from keras.optimizers import SGD
app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/Caicedo/Downloads/9+CodigoFuente/entrenamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


modelo = '1850131556'
ruta1 = 'C:/Users/Caicedo/Downloads/9+CodigoFuente/reconocimientofacial1/Data'

# # Start the infrared stream
# create keras model
batch_size = 32
img_height = 224
img_width = 224

pathDir = pathlib.Path(ruta1+'/train')
classnames = np.array([item.name for item in pathDir.glob('*') if item.name != "LICENSE.txt"])
print('classnames=', classnames)

def cut_faces():
    dataPath = 'C:/Users/Caicedo/Downloads/Recursos Humanos'
    if not os.path.exists('faces'):
        os.makedirs('faces')
    else:
        return

    for file in os.listdir(dataPath):
        if file.endswith('.jpg'):
            img = cv2.imread(dataPath+'/'+file)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite('faces/'+file, face)


cut_faces()

def createModel():
    dataPath = ruta1
    # read all images in the folder
    imagePaths = os.listdir(dataPath)
    print('imagePaths=', imagePaths)
    # create empty lists for both classes
    images = []
    classNum = []
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                               input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(classnames))
    ])
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True),
                    metrics=['accuracy'])
    model.summary()
    # load the images and get the class
    for imagePath in imagePaths:
        path = dataPath + '/' + imagePath
        for file in os.listdir(path):
            image = cv2.imread(path + '/' + file)
            image = cv2.resize(image, (img_height, img_width))
            images.append(image)
            classNum.append(imagePath)
    # encode the labels
    classes = np.unique(classNum)
    classNum = [np.where(classes == i)[0][0] for i in classNum]

    print('classes=', classes)
    # split the data into training and testing
    (trainX, testX, trainY, testY) = train_test_split(
        np.array(images), np.array(classNum), test_size=0.2)
    # train the model
    history = model.fit(
        trainX, trainY, batch_size=batch_size, epochs=10, validation_split=0.1)
    # save the model
    model.save('model.h5')
    print("Modelo guardado")



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
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
            x.append(img)
            y.append(i)
        i = i + 1

    return x, y, labels

modelLabels = dict()
def createRestNetModel():
    trainDir = ruta1+'/train'
    validDir = ruta1+'/val'
    # load images
    # trainX, trainY, modelLabels = loadImages(trainDir)
    # trainX = np.array(trainX)
    # trainY = np.array(trainY)
    # print('modelLabels=', modelLabels)
    # # convert class vectors to binary class matrices
    # trainY = keras.utils.to_categorical(trainY, modelLabels.__len__())
    # create the base pre-trained model
  
    # first: train only the top layers (which were randomly initialized)
    # # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                               input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(classnames), activation='softmax')
    ])

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # train the model on the new data for a few epochs
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        trainDir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        validDir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data= validation_generator,
       )

    # save the model
    model.save('model.h5')
    print("Modelo guardado")
    
   

# cargar el modelo si existe
if os.path.isfile('model.h5'):
    model = tf.keras.models.load_model('model.h5')
    print("Modelo cargado")
else:
    createRestNetModel()
    model = tf.keras.models.load_model('model.h5')
    print("Modelo cargado")

# openni2.initialize()
# capture = openni2.Device.open_any()
# # Start the depth stream
# depth_stream = capture.create_depth_stream()
# rgb_stream = capture.create_color_stream()
# depth_stream.start()
# rgb_stream.start()


def captureImage(identification):

    rutacompleta = ruta1 + '/train/' + identification
    rutaTest = ruta1 + '/val/' + identification
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)
    video = cv2.VideoCapture('http://192.168.137.72:4747/video')

    # Check if camera opened successfully
    if (video.isOpened() == False):
        print("Unable to read camera feed")
        return
    
    id = 0
    while True:
        ret, frame = video.read()
        # frame = depth_stream.read_frame()
        # frame_data = frame.get_buffer_as_uint16()
        # Get the frame from the color stream
        # frame1 = rgb_stream.read_frame()
        # frame_data1 = frame1.get_buffer_as_uint8()
        # Convert the frame data to an OpenCV image
        image1 = np.ndarray((frame.shape[0], frame.shape[1], 3), dtype=np.uint8, buffer=frame)
        cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        imageCopy = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image = np.ndarray((frame.height, frame.width),
        #                    dtype=np.uint16, buffer=frame_data)
        # image = np.uint8(image)

        faces = face_cascade.detectMultiScale(imageCopy, 1.3, 5)
        for (x, y, w, h) in faces:

            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rostroCapturado = imageCopy[y:y+h, x:x+w]
            # ndim 4 para que funcione con el modelo
            if(id>60):
                rostroCapturado = cv2.resize(
                    rostroCapturado, (224,224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(rutaTest + '/imagen_{}.jpg'.format(id),
                            rostroCapturado)
            else:
                rostroCapturado = cv2.resize(
                    rostroCapturado, (224,224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(rutacompleta + '/imagen_{}.jpg'.format(id),
                            rostroCapturado)
                
            id = id + 1
            x = int(x + w/2)
            y = int(y + h/2)
            # Mostrar el punto medio del rostro
            # cv2.circle(image1, (x, y), 2, (0, 0, 255), 2)
            # Mostrar la distancia del rostro
            if (x >= 240):
                x = 239
            if (y >= 320):
                y = 319
            # distance = image[x, y]
            # convertrir del rango de 0-255 a 100-3500
            # distance = distance * (3500-100)/255 + 100
            # solo 2 decimales
            # distance = round(distance, 2)
            # print(distance)
            # cv2.putText(image1, str(distance/10)+" cm", (x, y),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        # tomar 320 fotos y salir
        if id >= 80:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
    # verificamos si el modelo existe
    if os.path.isfile('model.h5'):
        # si existe cargamos el modelo
        model = tf.keras.models.load_model('model.h5')
        print("Modelo cargado")
    else:
        # si no existe creamos el modelo
        createRestNetModel()
        model = tf.keras.models.load_model('model.h5')
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
    video = cv2.VideoCapture('http://192.168.137.72:4747/video')
    id = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Convert the frame data to an OpenCV image
        image1 = np.ndarray((frame.shape[0], frame.shape[1], 3), dtype=np.uint8, buffer=frame)
        cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        imageCopy = image1.copy()

        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


        # frame = depth_stream.read_frame()
        # frame_data = frame.get_buffer_as_uint16()
        # # Get the frame from the color stream
        # frame1 = rgb_stream.read_frame()
        # frame_data1 = frame1.get_buffer_as_uint8()
        # # Convert the frame data to an OpenCV image
        # image1 = np.ndarray((frame1.height, frame1.width, 3),
        #                     dtype=np.uint8, buffer=frame_data1)
        # cv2.normalize(image1, image1, 0, 255, cv2.NORM_MINMAX)
        # color_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      

        # image = np.ndarray((frame.height, frame.width),
        #                    dtype=np.uint16, buffer=frame_data)
        # image = np.uint8(image)

        faces = face_cascade.detectMultiScale(imageCopy, 1.3, 5)
        for (x, y, w, h) in faces:
            
            face = imageCopy[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = np.ndarray.astype(face, np.float32)
            face = np.expand_dims(face, axis=0)
            
            face = preprocess_input(face)

            preds = model.predict(face)
            print(preds)
            i = preds.argmax(axis=1)[0]
            label = classnames[i]
            print(label)
            id = i
            # porcentaje de confianza
            confidence = preds[0][i]
            print(confidence)
            
            cv2.rectangle(image1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image1, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # obtener la distancia del centro del rostro
            x = x + w//2
            y = y + h//2
            if (x >= 240):
                x = 239
            if (y >= 320):
                y = 319
            # distance = image[x, y]
            # # convertrir del rango de 0-255 a 100-3500
            # distance = distance * (3500-100)/255 + 100
            # solo 2 decimales
            # distance = round(distance, 2)
            # print(distance)
            # if (distance < 1000):
            #     cv2.putText(image1, str(distance/10)+" cm", (x, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # cv2.putText(image1, str(distance/10)+" cm", (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            


        ret, buffer = cv2.imencode('.jpg', image1)
        frame2 = buffer.tobytes()
        if (id == 512):
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


def trainingModel():
    dataPath = ruta1
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
    print('Iniciando el entrenamiento...espere')
    trainingEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    trainingEigenFaceRecognizer.train(faceSamples, np.array(ids))
    trainingEigenFaceRecognizer.write('modelEigenFace.xml')
    print('Modelo EigenFace entrenado')
    entrenamientoEigenFaceRecognizer = cv2.face.EigenFaceRecognizer_create()
    entrenamientoEigenFaceRecognizer.read('modelEigenFace.xml')


def updateModel():
    dataPath = ruta1
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
    model = tf.keras.models.load_model('model.h5')
    model.fit(faceSamples, ids, epochs=10)
    model.save('model.h5')
