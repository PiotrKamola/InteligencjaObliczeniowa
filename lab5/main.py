import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import plot_model, to_categorical

import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

from tensorflow.keras.callbacks import History, ModelCheckpoint

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
random_state=42)
# Define the model
model = Sequential([
Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
Dense(64, activation='relu'),
Dense(y_encoded.shape[1], activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()
# Save the model
model.save('iris_model.h5')
# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

'''
a) Co robi StandardScaler? Jak transformowane są dane liczbowe?
b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? Jak etykiety klas są transformowane przez ten
encoder?
c) Model ma 4 warstwy: wejściową, dwie ukryte warstwy z 64 neuronami każda i warstwę wyjściową. Ile
neuronów ma warstwa wejściowa i co oznacza X_train.shape[1]? Ile neuronów ma warstwa wyjściowa i co
oznacza y_encoded.shape[1]?
d) Czy funkcja aktywacji relu jest najlepsza do tego zadania? Spróbuj użyć innej funkcji i obejrzyj wyniki
e) Model jest konfigurowany do treningu za pomocą polecenia compile. Tutaj wybieramy optymalizator (algorytm,
który używa gradientu straty do aktualizacji wag), funkcję straty, metrykę do oceny modelu. Eksperymentuj ze
zmianą tych parametrów na inne i uruchom program. Czy różne optymalizatory lub funkcje straty dają różne
wyniki? Czy możemy dostosować szybkość uczenia się w optymalizatorze?
f) W linii model.fit sieć neuronowa jest trenowana. Czy jest sposób, by zmodyfikować tę linię tak, aby rozmiar
partii był równy 4 lub 8 lub 16? Jak wyglądają krzywe uczenia się dla różnych parametrów? Jak zmiana partii
wpływa na kształt krzywych? Wypróbuj różne wartości i uruchom program.
g) Co możesz powiedzieć o wydajności sieci neuronowej na podstawie krzywych uczenia? W której epoce sieć
osiągnęła najlepszą wydajność? Czy ta krzywa sugeruje dobrze dopasowany model, czy mamy do czynienia z
niedouczeniem lub przeuczeniem?
h) Przejrzyj niżej wymieniony kod i wyjaśnij co się w nim dzieje.
'''

''' ####################################################################################################################
ZAD1

a) Standarizuje dane liczbowe, przekształca dane aby miały średnią 0 i odchylenie standardowe 1. 

b) One-Hot Encoding polega na reprezentowaniu etykiet klas w postaci wektora w którym tylko jeden bit jest ustawiony 
na 1, a reszta jest ustawiona na 0. Zamienia zmienne na format który może być użyty do algorytmów uczenia maszynowego 
w celu poprawy jego przewidywania. One-Hot Coding to kodowanie 1 z n. Zamiana reprezentacji danych na 0 i 1

c)  - Warstwa wejściowa ma tyle neuronów, ile cech w zbiorze danych wejściowych, czyli X_train.shape[1] - liczba kolumn 
w danych wejściowych.
    - Warstwa wyjściowa ma tyle neuronów, ile unikalnych klas w zbiorze etykiet, co odpowiada liczbie kolumn 
w y_encoded (y_encoded.shape[1]).

d)  - sigmoid najleszpa do zadań klasyfikacji binarnej
    - softmax stosowana w zadaniach klasyfikacji wieloklasowej
    - ReLU często stosowana w ukrytych warstwach sieci neuronowych 
    
    chyba (sigmoid relu softmax) najlepsze

e) done

f) done

g) chyba jest dobrze bo dokładność rośnie a strata maleje, najlepsza była epoka 80

h) 
    - wczytuje irisy
    - przekształcanie cechy aby miały średnią 0 i odchylenie standardowe 1 
    - przekształcanie klasy 0, 1, 2 w formę one-hot
    - train_test_split(X_scaled, y_encoded, test_size=0.3, dzieli dane na zbiór treningowy (70%) i testowy (30%)
    - random_state=42 gwarantuje powtarzalność losowania.
    - wczytuje wcześniej wytrenowany model z pliku iris_model.h5
    - kontynuuje trening dla 10 epok
    - zapisuje zaktualizowany model
    - ocena zaktualizowanego modelu

'''  ####################################################################################################################

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Load the pre-trained model
model = load_model('iris_model.h5')

###trzeba było jeszcze zrekompilować model###
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model for 10 more epochs
model.fit(X_train, y_train, epochs=10)

# Save the updated model
model.save('updated_iris_model.h5')

# Evaluate the updated model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

########################################################################################################################


# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Tworzenie callbacka do zapisywania najlepszego modelu
checkpoint = ModelCheckpoint(
    filepath='model_best_epoch.keras',  # Zmiana rozszerzenia na .keras
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Tworzenie modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
#     plt.xlabel(predicted_labels[i])
#     plt.show()

#wszystkie 25 na raz
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(f"Pred: {predicted_labels[i]}", fontsize=10)
plt.tight_layout()
plt.show()

''' ####################################################################################################################
ZAD2

a) W preprocessingu przygotowwane sa dane wejsciowe dla modelu
    - reshape przeksztalca macierz obrazow z wymiaru (28, 28) na (28, 28, 1), aby dodac wymiar kanalu 
    (konieczne dla warstw konwolucyjnych)
    - to_categorical zamienia etykiety numeryczne (np. 0, 1, 2) na macierze "one-hot" (np. 0 → [1, 0, 0, ...]).
    - np.argmax przeksztalca macierz "one-hot" z powrotem do etykiet numerycznych (np. [0, 1, 0, ...] → 1).

b)
    * Warstwa Conv2D (32 filtry 3x3, ReLU):
        - Wejscie: obraz 28x28x1. 
        - Wyjscie: mapa cech 26x26x32.
        - Kazdy filtr przetwarza fragmenty obrazu, uwzgledniajac wagi i funkcję aktywacji ReLU
    * MaxPooling2D (2x2):
        - Wejscie: mapa cech 26x26x32. 
        - Wyjscie: mapa cech 13x13x32.
        - Redukuje wymiary, wybierajac maksymalna wartosc w oknie 2x2
    * Flatten:
        - Wejscie: mapa cech 13x13x32. 
        - Wyjscie: wektor o rozmiarze 5408.
        - Rozplaszcza dane do jednowymiarowego wektora cech
    * Dense (64 jednostki, ReLU):
        - Wejscie: wektor 5408. 
        - Wyjscie: wektor 64.
        - Kazda jednostka oblicza kombinacje liniowa wag + funkcja ReLU.
    *Dense (10 jednostek, Softmax):
        - Wejscie: wektor 64. 
        - Wyjscie: wektor 10 (prawdopodobienstwo kazdej klasy).
        - Softmax przeksztalca dane na rozklad prawdopodobienstwa.

c) 
    - Ile: Przewidywana / Co bylo
    - 14: 3 / 5
    - 13: 4 / 9
    - 10: 2 / 7
    - 9: 8 / 9
    - 9: 3 / 9
    Najczesciej myli 3 z 5, 4 z 9 i 2 z 7

d) Raczej jest dobrze, nie ma przeuczenia (roznica miedzy dokladnoscia treningowa a walidacyjna jest mala) ani 
niedouczenia (dokładnosc walidacyjna jest wysoka (~98%), a strata walidacyjna jest niska (~0.05).)

e)
    dodac przed kompilacja:
    
    checkpoint = ModelCheckpoint(
        filepath='model_best_epoch.keras',  # Zmiana rozszerzenia na .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    teraz w porownaniu z podpunktem c:
    - Ile: Przewidywana / Co bylo
    - 14: 3 / 5     ->      6: 3 / 5
    - 13: 4 / 9     ->      12: 4 / 9
    - 10: 2 / 7     ->      2: 2 / 7
    - 9: 8 / 9      ->      1: 8 / 9
    - 9: 3 / 9      ->      6: 3 / 9
    
'''  ####################################################################################################################

# Z chatem GBT:

# import os
# import shutil
# from sklearn.model_selection import train_test_split
# import zipfile
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import History, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
# # Ścieżka do folderu z obrazkami
# image_dir = 'dogs-cats-mini'
#
# # Tworzenie struktury folderów
# train_dir = os.path.join(image_dir, 'train')
# validation_dir = os.path.join(image_dir, 'validation')
#
# os.makedirs(os.path.join(train_dir, 'cats'), exist_ok=True)
# os.makedirs(os.path.join(train_dir, 'dogs'), exist_ok=True)
# os.makedirs(os.path.join(validation_dir, 'cats'), exist_ok=True)
# os.makedirs(os.path.join(validation_dir, 'dogs'), exist_ok=True)
#
# # Pobranie wszystkich plików .jpg w folderze
# cat_images = [f for f in os.listdir(image_dir) if 'cat' in f and f.endswith('.jpg')]
# dog_images = [f for f in os.listdir(image_dir) if 'dog' in f and f.endswith('.jpg')]
#
# # Podział na zbiór treningowy i walidacyjny (80% do treningu, 20% do walidacji)
# train_cats, val_cats = train_test_split(cat_images, test_size=0.2, random_state=42)
# train_dogs, val_dogs = train_test_split(dog_images, test_size=0.2, random_state=42)
#
# # Przeniesienie plików do odpowiednich folderów
# for image in train_cats:
#     shutil.copy(os.path.join(image_dir, image), os.path.join(train_dir, 'cats', image))
# for image in val_cats:
#     shutil.copy(os.path.join(image_dir, image), os.path.join(validation_dir, 'cats', image))
#
# for image in train_dogs:
#     shutil.copy(os.path.join(image_dir, image), os.path.join(train_dir, 'dogs', image))
# for image in val_dogs:
#     shutil.copy(os.path.join(image_dir, image), os.path.join(validation_dir, 'dogs', image))
#
# # Inicjalizacja ImageDataGenerator do skalowania obrazów
# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# # Ładowanie danych
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     batch_size=32,
#     class_mode='binary',  # Dwie klasy: koty i psy
#     target_size=(150, 150)  # Rozmiar obrazków, np. 150x150
# )
#
# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     batch_size=32,
#     class_mode='binary',
#     target_size=(150, 150)
# )
#
# # Krok 1: Przygotowanie danych
#
# # Wypakowanie pliku ZIP
# # zip_file = "dogs-cats-mini.zip"
# extract_folder = "dogs-cats-mini"
# # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
# #     zip_ref.extractall(extract_folder)
#
# # Załaduj dane przy użyciu ImageDataGenerator
# train_dir = os.path.join(extract_folder, 'train')
# validation_dir = os.path.join(extract_folder, 'validation')
#
# # ImageDataGenerator do przetwarzania danych (skalowanie, augmentacja)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# # Generatory danych
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary'  # klasyfikacja binarna (kot/pies)
# )
#
# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='binary'
# )
#
# # Krok 2: Budowanie modelu
#
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Funkcja aktywacji sigmoid dla klasyfikacji binarnej
# ])
#
# # Kompilacja modelu
# model.compile(
#     optimizer=Adam(),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
#
# # Krok 3: Trenowanie modelu z callbackiem do zapisywania najlepszego modelu
# checkpoint = ModelCheckpoint(
#     filepath='best_model.keras',
#     monitor='val_accuracy',
#     save_best_only=True,
#     mode='max',
#     verbose=1
# )
#
# history = History()
#
# # Trenowanie modelu
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     callbacks=[checkpoint, history]
# )
#
# # Krok 4: Walidacja i analiza wyników
#
# # Ocena modelu
# test_loss, test_acc = model.evaluate(validation_generator)
# print(f"Test accuracy: {test_acc:.4f}")
#
# # Krzywe uczenia się
# plt.figure(figsize=(10, 5))
#
# # Krzywa dokładności
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
#
# # Krzywa straty
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# import os
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from numpy import asarray, save
#
# # 1. Ustawienia
# folder = 'test/'  # Folder z obrazami
# target_size = (50, 50)  # Rozmiar docelowy obrazów
#
# # 2. Inicjalizacja list na dane
# photos, labels = [], []
#
# # 3. Wczytywanie obrazów i etykiet
# print("Wczytywanie danych...")
# for file in os.listdir(folder):
#     # Klasa: 1.0 dla psa, 0.0 dla kota
#     output = 1.0 if file.startswith('dog') else 0.0
#
#     # Wczytaj obraz, zmień rozmiar i zamień na tablicę NumPy
#     photo = load_img(os.path.join(folder, file), target_size=target_size)
#     photo = img_to_array(photo)
#
#     # Dodaj dane do list
#     photos.append(photo)
#     labels.append(output)
#
# # 4. Konwersja danych do tablic NumPy
# photos = asarray(photos)
# labels = asarray(labels)
#
# print(f"Wczytano dane: {photos.shape} obrazy, {labels.shape} etykiety.")
#
# # 5. Zapis do plików .npy
# print("Zapisywanie danych...")
# save('dogs_vs_cats_photos.npy', photos)
# save('dogs_vs_cats_labels.npy', labels)
# print("Zapis zakończony.")


# # load and confirm the shape
# from numpy import load
# photos = load('dogs_vs_cats_photos.npy')
# labels = load('dogs_vs_cats_labels.npy')
# print(photos.shape, labels.shape)

from os import makedirs, listdir
from shutil import copyfile
from numpy.random import seed, random

# # create directories
# dataset_home = 'dataset_dogs_vs_cats/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
# 	# create label subdirectories
# 	labeldirs = ['dogs/', 'cats/']
# 	for labldir in labeldirs:
# 		newdir = dataset_home + subdir + labldir
# 		makedirs(newdir, exist_ok=True)
#
#
#     # seed random number generator
# seed(1)
# # define ratio of pictures to use for validation
# val_ratio = 0.25
# # copy training dataset images into subdirectories
# src_directory = 'train/'
# for file in listdir(src_directory):
#     src = src_directory + '/' + file
#     dst_dir = 'train/'
#     if random() < val_ratio:
#         dst_dir = 'test/'
#     if file.startswith('cat'):
#         dst = dataset_home + dst_dir + 'cats/' + file
#         copyfile(src, dst)
#     elif file.startswith('dog'):
#         dst = dataset_home + dst_dir + 'dogs/' + file
#         copyfile(src, dst)

# baseline model for the dogs vs cats dataset


# import os
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from numpy import asarray, save
#
#
# # Funkcja do przetwarzania folderu (train/test)
# def process_folder(base_folder, target_size=(100, 100)):
# 	photos, labels = [], []
# 	classes = {'cats': 0, 'dogs': 1}
#
# 	# Iteracja przez foldery 'cats' i 'dogs'
# 	for label, class_value in classes.items():
# 		class_folder = os.path.join(base_folder, label)
# 		for file in os.listdir(class_folder):
# 			file_path = os.path.join(class_folder, file)
#
# 			# Wczytanie obrazu, zmiana rozmiaru i konwersja do NumPy
# 			image = load_img(file_path, target_size=target_size)
# 			image = img_to_array(image)
#
# 			photos.append(image)
# 			labels.append(class_value)
#
# 	return asarray(photos), asarray(labels)
#
#
# # Ścieżki do folderów
# dataset_path = 'dataset_dogs_vs_cats'
# train_path = os.path.join(dataset_path, 'train')
# test_path = os.path.join(dataset_path, 'test')
#
# # Przetwarzanie danych
# print("Przetwarzanie danych treningowych...")
# train_photos, train_labels = process_folder(train_path)
# print(f"Zapisano {train_photos.shape[0]} obrazów treningowych.")
#
# print("Przetwarzanie danych testowych...")
# test_photos, test_labels = process_folder(test_path)
# print(f"Zapisano {test_photos.shape[0]} obrazów testowych.")
#
# # Zapisywanie do plików .npy
# print("Zapisywanie do plików .npy...")
# save('train_photos.npy', train_photos)
# save('train_labels.npy', train_labels)
# save('test_photos.npy', test_photos)
# save('test_labels.npy', test_labels)
#
# print("Proces zakończony!")

# from numpy import load
#
# # Odczyt
# train_photos = load('train_photos.npy')
# train_labels = load('train_labels.npy')
# test_photos = load('test_photos.npy')
# test_labels = load('test_labels.npy')
#
# print(f"Dane treningowe: {train_photos.shape}, Etykiety: {train_labels.shape}")
# print(f"Dane testowe: {test_photos.shape}, Etykiety: {test_labels.shape}")

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# # Załadowanie danych
# train_photos = np.load('train_photos.npy')
# train_labels = np.load('train_labels.npy')
# test_photos = np.load('test_photos.npy')
# test_labels = np.load('test_labels.npy')
#
# # Normalizacja danych (przemiana wartości pikseli do zakresu [0, 1])
# train_photos = train_photos.astype('float32') / 255.0
# test_photos = test_photos.astype('float32') / 255.0
#
# # Kodowanie etykiet (w przypadku klasyfikacji binarnej)
# train_labels = to_categorical(train_labels, 2)
# test_labels = to_categorical(test_labels, 2)
# # Tworzenie modelu CNN
# model = Sequential()
#
# # Pierwsza warstwa konwolucyjna
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Druga warstwa konwolucyjna
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Trzecia warstwa konwolucyjna
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # Rozwinięcie do jednej warstwy w pełni połączonej
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
#
# # Warstwa wyjściowa (softmax dla klasyfikacji binarnej)
# model.add(Dense(2, activation='softmax'))
# # Kompilacja modelu
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# # Trenowanie modelu
# history = model.fit(train_photos, train_labels,
#                     epochs=10, batch_size=32,
#                     validation_data=(test_photos, test_labels))
# # Ocena modelu na zbiorze testowym
# test_loss, test_acc = model.evaluate(test_photos, test_labels)
# print(f"Test Accuracy: {test_acc:.4f}")
# # Zapisanie modelu
# model.save('dogs_vs_cats_model.h5')
# import matplotlib.pyplot as plt
#
# # Wizualizacja wyników
# plt.plot(history.history['accuracy'], label='Dokładność treningowa')
# plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
# plt.title('Dokładność modelu')
# plt.xlabel('Epoki')
# plt.ylabel('Dokładność')
# plt.legend()
# plt.show()
