from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import  tensorflow.keras.utils as utilsTF
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


diseaseType = input("Enter Disease Type: ")
if diseaseType == "malaria":
    warnings.filterwarnings("ignore")
    classifier = Sequential()
    classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Convolution2D(64,3,3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(
            './dataset/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

    test_set = test_datagen.flow_from_directory(
            './dataset/testing_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')
    classifier.fit_generator(
            training_set,
            steps_per_epoch=5000,
            epochs=10,
            validation_data=test_set,
            validation_steps =1000)

    test_image = utilsTF.load_img("C:\\Users\\Tawfiq\\Downloads\\UARS5y85RKUs6m2bo1duw.png", target_size = (64, 64))
    test_image = utilsTF.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'Uninfected with malaria'
    else:
        prediction = 'Parasitised malaria'
    print(prediction)
elif diseaseType == "parkinson":
    parkinsons_data = pd.read_csv('./parkinsons.data')
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
    input_data = (
    197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680,
    0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        print("The Person does not have Parkinsons Disease\n\n")
    else:
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
