import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Funkcja do porównania metod konwersji
def compare_grayscale_methods(image_path):
    # Wczytanie obrazu w formacie BGR
    image = cv2.imread(image_path)

    if image is None:
        print(f"Nie udało się wczytać obrazu: {image_path}")
        return

    # Konwersja na skalę szarości (średnia arytmetyczna)
    gray_simple = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Konwersja na skalę szarości (ważona średnia)
    weighted_gray = np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

    # Wyświetlenie obrazów
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Oryginalny obraz")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray_simple, cmap='gray')
    plt.title("Skala szarości (średnia)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(weighted_gray, cmap='gray')
    plt.title("Skala szarości (ważona)")
    plt.axis("off")

    plt.show()


# Ścieżka do folderu z obrazami
folder_path = "bird_miniatures"

# Iteracja po wszystkich plikach w folderze
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    # Sprawdzenie, czy plik to obraz
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Przetwarzanie: {filename}")
        compare_grayscale_methods(image_path)
