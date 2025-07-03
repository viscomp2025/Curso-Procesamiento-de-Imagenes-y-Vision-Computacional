import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

  # Cargar y convertir a escala de grises  
img = cv2.imread('monedas.jpg')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

  # Umbralización manual  
_, thresh_manual = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

  # Umbralización Otsu  
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

  # Umbralización adaptativa  
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
                                          cv2.THRESH_BINARY, 11, 2)  

  # Visualización de resultados  
plt.figure(figsize=(12, 8))  
plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')  
plt.subplot(222), plt.imshow(thresh_manual, cmap='gray'), plt.title('Umbral Manual (127)')  
plt.subplot(223), plt.imshow(thresh_otsu, cmap='gray'), plt.title('Umbral Otsu')  
plt.subplot(224), plt.imshow(thresh_adaptive, cmap='gray'), plt.title('Umbral Adaptativo')  
plt.tight_layout()  
plt.show()  

  # Guardar imágenes  
cv2.imwrite('thresh_manual.jpg', thresh_manual)  
cv2.imwrite('thresh_otsu.jpg', thresh_otsu)  
cv2.imwrite('thresh_adaptive.jpg', thresh_adaptive)
