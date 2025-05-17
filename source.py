from skimage import data
import cv2
import os
from PIL import Image
import numpy as np

def con_img_gray_scale():
    path = "img/in"
    output_path = "img/out"
    if not os.path.isdir(path):
        print(f"Błąd: Folder {path} nie istnieje.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(path):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            file_path = os.path.join(path,filename)
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)   # Skala szarosci
                if img is None:
                    raise ValueError("Nie udało wczytać obrazu")
                
                
                # Proces segmentacji
                seg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Binaryzacja (Progowanie Otsu)
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Ściemnianie
                dark = cv2.convertScaleAbs(img, alpha=0.5, beta=0) # alpha zmniejsza jasność o 50%

                # Zapisanie przetworzonych obrazów
                cv2.imwrite(os.path.join(output_path, f"segmented_{filename}"), seg)
                cv2.imwrite(os.path.join(output_path, f"binary_{filename}"), binary)
                cv2.imwrite(os.path.join(output_path, f"dark_{filename}"), dark)

            except Exception as e:
                print(f"Uszkodzony plik: {filename}\t| Błąd {e}")

if __name__ == "__main__":
    con_img_gray_scale()