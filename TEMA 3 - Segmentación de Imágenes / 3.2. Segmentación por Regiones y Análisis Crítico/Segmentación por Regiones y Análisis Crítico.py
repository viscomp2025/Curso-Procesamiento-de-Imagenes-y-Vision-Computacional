import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

  # Lista de imágenes  
images = ['luz_uniforme.jpg', 'sombra_parcial.jpg', 'tornillos.jpg']  
results = {}  

for img_name in images:  
      # Cargar y convertir a escala de grises  
      img = cv2.imread(img_name)  
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

      # Umbralizaciones  
      _, thresh_manual_100 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  
      _, thresh_manual_150 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  
      _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
      thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
                                              cv2.THRESH_BINARY, 9, 2)  

      # Segmentación por watershed (ejemplo simplificado)  
      markers = np.zeros(gray.shape, dtype=np.int32)  
      cv2.watershed(img, markers)  
      watershed = cv2.convertScaleAbs(markers)  

      # Detectar contornos  
      contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
      img_contours = img.copy()  
      cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)  
      num_contours = len(contours)  

      # Guardar resultados  
      results[img_name] = {'contours': num_contours, 'image': img_contours}  
      cv2.imwrite(f'{img_name}_watershed.jpg', watershed)  
      cv2.imwrite(f'{img_name}_contours.jpg', img_contours)  

  # Visualización de una imagen de ejemplo  
plt.figure(figsize=(10, 5))  
plt.subplot(121), plt.imshow(cv2.cvtColor(results[images[0]]['image'], cv2.COLOR_BGR2RGB)), plt.title(f'{images[0]} - Contornos: {results[images[0]]["contours"]}')  
plt.subplot(122), plt.imshow(watershed, cmap='gray'), plt.title(f'{images[0]} - Watershed')  
plt.tight_layout()  
plt.show()

