from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow.keras.utils as utilsTF
import warnings
import numpy as np
import threading
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.label import MDLabel
from kivy.lang import Builder
from kivymd.uix.button import MDRoundFlatButton
import os
import plyer

DataField = '''
MDTextField:
    hint_text: "Enter Data to classify: "
    helper_text: "Enter Patient Data"
    helper_text_mode: "on_focus"
    pos_hint: {'center_x':0.5,'center_y':0.65}
    required:True
    size_hint_x:None
    width:300
'''

class DiseasesDetector(MDApp):
    def build(self):
        self.screen = Screen()
        self.header_l = MDLabel(text="DoctorX", pos_hint={"center_y": .9})
        self.header_l.font_style = "H2"
        self.imagePath = ""
        self.header_l.halign = "center"
        self.realPath = os.getcwd()
        self.parkinsonU = MDRoundFlatButton(text="Parkinson", pos_hint={"center_x": .5, "center_y": .55}, width=.9,
                                      size_hint=(.5, .07),on_press = self.ParkisnonUI)
        self.malaria = MDRoundFlatButton(text="Malaria", pos_hint={"center_x": .5, "center_y": .45}, width=.9,
                                    size_hint=(.5, .07),on_press = self.malariaUI)
        self.screen.add_widget(self.header_l)
        self.screen.add_widget(self.parkinsonU)
        self.screen.add_widget(self.malaria)
        return self.screen

    def malariaDisease(self, pathImage):
        try:
            self.ResultDataMalaria.text = "Started The processing"
            warnings.filterwarnings("ignore")
            classifier = Sequential()
            classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Convolution2D(64, 3, 3, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Flatten())
            classifier.add(Dense(128, activation='relu'))
            classifier.add(Dense(1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            from keras.preprocessing.image import ImageDataGenerator
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            training_set = train_datagen.flow_from_directory(
                self.realPath+'\\dataset\\training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

            test_set = test_datagen.flow_from_directory(
                self.realPath+'\\dataset\\testing_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')
            classifier.fit_generator(
                training_set,
                steps_per_epoch=500,
                epochs=5,
                validation_data=test_set,
                validation_steps=100)

            test_image = utilsTF.load_img(pathImage, target_size=(64, 64))
            test_image = utilsTF.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict(test_image)
            training_set.class_indices
            if result[0][0] == 1:
                prediction = 'Uninfected with malaria'
            else:
                prediction = 'Parasitised malaria'
            print(prediction)
            self.ResultDataMalaria.text = prediction
        except Exception as error:
            self.ResultDataMalaria.text = "Error occured. check that the image is correct"
            print(error)

    def parkinson(self):
        test = tuple(float(x) for x in self.inputData.text.split(","))
        print(test)
        self.ResultDataParkinson.text = "started.."
        time.sleep(2)
        parkinsons_data = pd.read_csv(self.realPath + '/parkinsons.data')
        parkinsons_data.head()
        parkinsons_data.shape
        parkinsons_data.info()
        parkinsons_data.isnull().sum()
        parkinsons_data.describe()
        parkinsons_data['status'].value_counts()
        parkinsons_data.groupby('status').mean()
        X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
        Y = parkinsons_data['status']
        print(Y)
        self.ResultDataParkinson.text = "Y: {}".format(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        print(X.shape, X_train.shape, X_test.shape)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print(X_train)
        model = svm.SVC(kernel='linear')
        model.fit(X_train, Y_train)
        X_train_prediction = model.predict(X_train)

        training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
        print('Accuracy score of training data : ', training_data_accuracy)
        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
        print('Accuracy score of test data : ', test_data_accuracy)
        input_data = test
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = model.predict(std_data)
        print(prediction)

        if (prediction[0] == 0):
            print("The Person does not have Parkinsons Disease\n\n")
            self.ResultDataParkinson.text = "The Person does not have Parkinsons Disease\nAccuracy score of test data : " + str(test_data_accuracy) + "\n" + 'Accuracy score of training data : ' + str(training_data_accuracy)
        else:
            self.ResultDataParkinson.text = "The Person has Parkinsons\nAccuracy score of test data : " + str(
                test_data_accuracy) + "\n" + 'Accuracy score of training data : ' + str(training_data_accuracy)
            print("The Person has Parkinsons")
        with open("dataFile.txt", "a", encoding="utf-8") as filo:
            filo.write("x: " + str(X) + "\n")
            filo.write("y: " + str(Y) + "\n")
            filo.write("x train: " + str(X_train) + "\n")
            filo.write(
                "x Shape: " + str(X.shape) + "\n" + "Train Shape :" + str(X_train.shape) + "\n" + "Test Shape: " + str(
                    X_test.shape) + "\n")
            filo.write('Accuracy score of test data : ' + str(test_data_accuracy) + "\n")
            filo.write('Accuracy score of training data : ' + str(training_data_accuracy) + "\n")
    def malariaUI(self,*args):
        self.screen.clear_widgets()
        headerMALARIA = MDLabel(text = "Malaria Disease", pos_hint = {"center_y":.8},font_style = "H2")
        headerMALARIA.halign = "center"
        result_Detect = MDRoundFlatButton(text = "Result",pos_hint = {"center_x":.5,"center_y":.2},size_hint=(.5, .07),on_press = self.startDetectMalaria)
        filechoose = MDRoundFlatButton(text="getImage", pos_hint={"center_x": .5, "center_y": .6}, size_hint=(.5, .07),on_press = self.getImagePath)
        Return_Detect = MDRoundFlatButton(text="Return Back", pos_hint={"center_x": .5, "center_y": .35},size_hint=(.5, .07),on_press = self.returnStart)
        self.ResultDataMalaria = MDLabel(text="Press the button to get the result", pos_hint={"center_y": .5})
        self.ResultDataMalaria.halign = "center"
        self.screen.add_widget(headerMALARIA)
        self.screen.add_widget(result_Detect)
        self.screen.add_widget(Return_Detect)
        self.screen.add_widget(self.ResultDataMalaria)
        self.screen.add_widget(filechoose)
    def returnStart(self,*args):
        self.screen.clear_widgets()
        self.screen.add_widget(self.header_l)
        self.screen.add_widget(self.parkinsonU)
        self.screen.add_widget(self.malaria)
    def getImagePath(self,*args):
        plyer.filechooser.open_file(on_selection=self.handle_selection)
    def handle_selection(self, selection):
        self.imagePath = selection
        self.ResultDataMalaria.text = "got file path"
    def startDetectMalaria(self,*args):
        malariaThread = threading.Thread(target=self.malariaDisease,args = (self.imagePath))
        malariaThread.start()
    def ParkisnonUI(self,*args):
        self.screen.clear_widgets()
        headerParkinson = MDLabel(text="Parkinson Disease", pos_hint={"center_y": .85}, font_style="H2")
        Return_Detect = MDRoundFlatButton(text="Return Back", pos_hint={"center_x": .5, "center_y": .1},
                                          size_hint=(.5, .07), on_press=self.returnStart)
        result_Detect = MDRoundFlatButton(text="Detect Parkinson", pos_hint={"center_x": .5, "center_y": .2}, size_hint=(.5, .07),on_press = self.SolveParkinson)
        headerParkinson.halign = "center"
        self.ResultDataParkinson = MDLabel(text="Press the button to get the result", pos_hint={"center_y": .32})
        self.ResultDataParkinson.halign = "center"
        Understand = MDRoundFlatButton(text="Understand Data", pos_hint={"center_x": .5, "center_y": .5},
                                          size_hint=(.5, .07), on_press=self.underSTandData)
        self.inputData = Builder.load_string(DataField)
        self.screen.add_widget(headerParkinson)
        self.screen.add_widget(Return_Detect)
        self.screen.add_widget(result_Detect)
        self.screen.add_widget(Understand)
        self.screen.add_widget(self.ResultDataParkinson)
        self.screen.add_widget(self.inputData)
    def underSTandData(self,*args):
        self.screen.clear_widgets()
        self.infoHeader = MDLabel(text = "DataInfo",pos_hint = {"center_y":.9},font_style = "H2")
        self.infoHeader.halign = "center"
        self.INFOParkinson = MDLabel(
            text="Data:\n(MDVP:Fo(Hz)\nMDVP:Fhi(Hz)\nMDVP:Flo(Hz)\nMDVP:Jitter(%)\nMDVP:Jitter(Abs)\nMDVP:RAP\nMDVP:PPQ\nJitter:DDP\nMDVP:Shimmer\nMDVP:Shimmer(dB)\nShimmer:APQ3\nShimmer:APQ5\nMDVP:APQ\nShimmer:DDA\nNHR, HNR, RPDE, DFA\n spread1\nspread2\nD2, PPE)",
            pos_hint={"center_y": .55})
        Return_Detect = MDRoundFlatButton(text="Return Back", pos_hint={"center_x": .5, "center_y": .1},
                                          size_hint=(.5, .07), on_press=self.ParkisnonUI)
        self.INFOParkinson.halign = "center"
        self.infoHeader2 = MDLabel(text="You will write them comma separated numbers.", pos_hint={"center_y": .2})
        self.infoHeader2.halign = "center"
        self.screen.add_widget(self.INFOParkinson)
        self.screen.add_widget(self.infoHeader)
        self.screen.add_widget(self.infoHeader2)
        self.screen.add_widget(Return_Detect)
    def SolveParkinson(self,*args):
        threadParkinson = threading.Thread(target=self.parkinson)
        threadParkinson.start()

DiseasesDetector().run()
#119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654