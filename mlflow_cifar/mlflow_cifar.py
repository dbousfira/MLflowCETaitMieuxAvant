import sys
import time

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
from urllib.parse import urlparse


DATA_ROOT = "../data/"
TRAINING_PATH = DATA_ROOT + "cifar_lite/train"
TESTING_PATH = DATA_ROOT + "cifar_lite/test"
img_size = 32
epochs = 50

filter = int(sys.argv[1]) if len(sys.argv) > 1 else 50

with mlflow.start_run():
    # Build the model
    tic = time.time()
    model = Sequential()

    model.add(
        Conv2D(
            filter,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3)))
    model.add(
        Conv2D(
            125,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='adam')

    mlflow.keras.autolog()

    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    # Convertion des futurs images RBG en niveaux de gris
    # (0 == noir, 255 == blanc)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_PATH,
        target_size=(img_size, img_size),
        class_mode="categorical")

    test_generator = test_datagen.flow_from_directory(
        directory=TESTING_PATH,
        target_size=(img_size, img_size),
        class_mode="categorical")
    entrainement = model.fit_generator(
        train_generator,
        epochs=epochs,
        callbacks=[es_callback],
        validation_data=test_generator)
    duration_training = time.time() - tic

    score = model.evaluate(train_generator, verbose=0)
    print("Test de perte:", score[0])
    print("Test de précision:", score[1])

    print("CNN model (loss=%f, accuracy=%f):" % (score[0], score[1]))

    # Evaluate the model prediction
    metrics = {
        "duration_training": duration_training,
    }

    # Log in mlflow (metrics)
    mlflow.log_metrics(metrics)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.keras.log_model(
            model, "model", registered_model_name="CNNModel")
    else:
        mlflow.keras.log_model(model, "model")
