import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def count_and_display_birds(image_path):
    # Wczytanie obrazu w skali szarości
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Nie udało się wczytać obrazu: {image_path}")
        return 0

    # Rozmycie, aby zredukować szum
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Progowanie adaptacyjne
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Znalezienie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtracja konturów na podstawie wielkości (eliminacja szumu)
    min_contour_area = 1
    bird_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Liczba ptaków
    bird_count = len(bird_contours)

    # Rysowanie konturów na oryginalnym obrazie
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, bird_contours, -1, (0, 255, 0), 2)

    # Wyświetlenie wyników
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Oryginalny obraz")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Obraz po progowaniu")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(output_image)
    plt.title(f"Liczba ptaków: {bird_count}")
    plt.axis("off")

    plt.show()

    return bird_count

# Funkcja do przetwarzania wszystkich obrazów w folderze
def process_and_display_all_images(folder_path):
    # Pobranie listy plików w folderze
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = {}

    # Iteracja po każdym pliku
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        bird_count = count_and_display_birds(image_path)
        results[image_file] = bird_count
        print(f"Liczba ptaków na obrazie {image_file}: {bird_count}")

folder_path = "bird_miniatures"
process_and_display_all_images(folder_path)