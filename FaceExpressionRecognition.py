import keras,os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

fit = False #code flow control variable
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
model = Sequential()

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

#if you wish to train the model
if fit:


    batch_size = 256
    epochs = 5
    #use fer2013 dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    with open(os.getcwd()+"/data/fer2013.csv") as f:
        content = f.readlines()
    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ", num_of_instances)

    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")

            val = img.split(" ")
            pixels = np.array(val, 'float32')

            emotion = keras.utils.to_categorical(emotion, num_classes)

            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("", end="")

    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
    )

    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', 100*train_score[1])

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100*test_score[1])

    model.save_weights('configs/my_model_weights.h5')
else:
    #import weights from file and use the model without training
    model.load_weights('configs/my_model_weights.h5')


#test with a image that is inside images/ directory
img = image.load_img(os.getcwd()+"/images/raiva.jpeg", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
plt.figure(1)
plt.subplot(211)
emotion_analysis(custom[0])
x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.subplot(212)
plt.imshow(x)
plt.show()
