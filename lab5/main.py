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

'''
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


# a) Co robi StandardScaler? Jak transformowane są dane liczbowe?
# b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? Jak etykiety klas są transformowane przez ten
# encoder?
# c) Model ma 4 warstwy: wejściową, dwie ukryte warstwy z 64 neuronami każda i warstwę wyjściową. Ile
# neuronów ma warstwa wejściowa i co oznacza X_train.shape[1]? Ile neuronów ma warstwa wyjściowa i co
# oznacza y_encoded.shape[1]?
# d) Czy funkcja aktywacji relu jest najlepsza do tego zadania? Spróbuj użyć innej funkcji i obejrzyj wyniki
# e) Model jest konfigurowany do treningu za pomocą polecenia compile. Tutaj wybieramy optymalizator (algorytm,
# który używa gradientu straty do aktualizacji wag), funkcję straty, metrykę do oceny modelu. Eksperymentuj ze
# zmianą tych parametrów na inne i uruchom program. Czy różne optymalizatory lub funkcje straty dają różne
# wyniki? Czy możemy dostosować szybkość uczenia się w optymalizatorze?
# f) W linii model.fit sieć neuronowa jest trenowana. Czy jest sposób, by zmodyfikować tę linię tak, aby rozmiar
# partii był równy 4 lub 8 lub 16? Jak wyglądają krzywe uczenia się dla różnych parametrów? Jak zmiana partii
# wpływa na kształt krzywych? Wypróbuj różne wartości i uruchom program.
# g) Co możesz powiedzieć o wydajności sieci neuronowej na podstawie krzywych uczenia? W której epoce sieć
# osiągnęła najlepszą wydajność? Czy ta krzywa sugeruje dobrze dopasowany model, czy mamy do czynienia z
# niedouczeniem lub przeuczeniem?
# h) Przejrzyj niżej wymieniony kod i wyjaśnij co się w nim dzieje.


########################################################################################################################
# ZAD1
# 
# a) Standarizuje dane liczbowe, przekształca dane aby miały średnią 0 i odchylenie standardowe 1. 
# 
# b) One-Hot Encoding polega na reprezentowaniu etykiet klas w postaci wektora w którym tylko jeden bit jest ustawiony 
# na 1, a reszta jest ustawiona na 0. Zamienia zmienne na format który może być użyty do algorytmów uczenia maszynowego 
# w celu poprawy jego przewidywania. One-Hot Coding to kodowanie 1 z n. Zamiana reprezentacji danych na 0 i 1
# 
# c)  - Warstwa wejściowa ma tyle neuronów, ile cech w zbiorze danych wejściowych, czyli X_train.shape[1] - liczba kolumn 
# w danych wejściowych.
#     - Warstwa wyjściowa ma tyle neuronów, ile unikalnych klas w zbiorze etykiet, co odpowiada liczbie kolumn 
# w y_encoded (y_encoded.shape[1]).
# 
# d)  - sigmoid najleszpa do zadań klasyfikacji binarnej
#     - softmax stosowana w zadaniach klasyfikacji wieloklasowej
#     - ReLU często stosowana w ukrytych warstwach sieci neuronowych 
#     
#     chyba (sigmoid relu softmax) najlepsze
# 
# e) done
# 
# f) done
# 
# g) chyba jest dobrze bo dokładność rośnie a strata maleje, najlepsza była epoka 80
# 
# h) 
#     - wczytuje irisy
#     - przekształcanie cechy aby miały średnią 0 i odchylenie standardowe 1 
#     - przekształcanie klasy 0, 1, 2 w formę one-hot
#     - train_test_split(X_scaled, y_encoded, test_size=0.3, dzieli dane na zbiór treningowy (70%) i testowy (30%)
#     - random_state=42 gwarantuje powtarzalność losowania.
#     - wczytuje wcześniej wytrenowany model z pliku iris_model.h5
#     - kontynuuje trening dla 10 epok
#     - zapisuje zaktualizowany model
#     - ocena zaktualizowanego modelu

########################################################################################################################

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


########################################################################################################################
# ZAD2
# 
# a) W preprocessingu przygotowwane sa dane wejsciowe dla modelu
#     - reshape przeksztalca macierz obrazow z wymiaru (28, 28) na (28, 28, 1), aby dodac wymiar kanalu 
#     (konieczne dla warstw konwolucyjnych)
#     - to_categorical zamienia etykiety numeryczne (np. 0, 1, 2) na macierze "one-hot" (np. 0 → [1, 0, 0, ...]).
#     - np.argmax przeksztalca macierz "one-hot" z powrotem do etykiet numerycznych (np. [0, 1, 0, ...] → 1).
# 
# b)
#     * Warstwa Conv2D (32 filtry 3x3, ReLU):
#         - Wejscie: obraz 28x28x1. 
#         - Wyjscie: mapa cech 26x26x32.
#         - Kazdy filtr przetwarza fragmenty obrazu, uwzgledniajac wagi i funkcję aktywacji ReLU
#     * MaxPooling2D (2x2):
#         - Wejscie: mapa cech 26x26x32. 
#         - Wyjscie: mapa cech 13x13x32.
#         - Redukuje wymiary, wybierajac maksymalna wartosc w oknie 2x2
#     * Flatten:
#         - Wejscie: mapa cech 13x13x32. 
#         - Wyjscie: wektor o rozmiarze 5408.
#         - Rozplaszcza dane do jednowymiarowego wektora cech
#     * Dense (64 jednostki, ReLU):
#         - Wejscie: wektor 5408. 
#         - Wyjscie: wektor 64.
#         - Kazda jednostka oblicza kombinacje liniowa wag + funkcja ReLU.
#     *Dense (10 jednostek, Softmax):
#         - Wejscie: wektor 64. 
#         - Wyjscie: wektor 10 (prawdopodobienstwo kazdej klasy).
#         - Softmax przeksztalca dane na rozklad prawdopodobienstwa.
# 
# c) 
#     - Ile: Przewidywana / Co bylo
#     - 14: 3 / 5
#     - 13: 4 / 9
#     - 10: 2 / 7
#     - 9: 8 / 9
#     - 9: 3 / 9
#     Najczesciej myli 3 z 5, 4 z 9 i 2 z 7
# 
# d) Raczej jest dobrze, nie ma przeuczenia (roznica miedzy dokladnoscia treningowa a walidacyjna jest mala) ani 
# niedouczenia (dokładnosc walidacyjna jest wysoka (~98%), a strata walidacyjna jest niska (~0.05).)
# 
# e)
#     dodac przed kompilacja:
#     
#     checkpoint = ModelCheckpoint(
#         filepath='model_best_epoch.keras',  # Zmiana rozszerzenia na .keras
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     )
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(10, activation='softmax')
#     ])
#     
#     teraz w porownaniu z podpunktem c:
#     - Ile: Przewidywana / Co bylo
#     - 14: 3 / 5     ->      6: 3 / 5
#     - 13: 4 / 9     ->      12: 4 / 9
#     - 10: 2 / 7     ->      2: 2 / 7
#     - 9: 8 / 9      ->      1: 8 / 9
#     - 9: 3 / 9      ->      6: 3 / 9
    
########################################################################################################################
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

data_dir = "dogs-cats-mini"
image_size = (150, 150)

# Funkcja ładowania obrazów z folderu
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
        # Etykieta na podstawie nazwy pliku
        label = 1 if 'dog' in filename.lower() else 0
        labels.append(label)
    return images, labels

# Ładowanie wszystkich obrazów
all_images, all_labels = load_images_from_folder(data_dir)

# Przekształcenie do formatu numpy
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Podział na zbiory treningowy, walidacyjny i testowy
X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalizacja obrazów
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Model sieci
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Wizualizacja wyników
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Ewaluacja na zbiorze testowym
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predykcja na zbiorze testowym
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
print("Macierz pomyłek:")
print(cm)

# Raport klasyfikacji
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

