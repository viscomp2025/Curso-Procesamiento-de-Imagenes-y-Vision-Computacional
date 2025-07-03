import cv2  
  import numpy as np  
  import matplotlib.pyplot as plt  

  # Cargar imagen propia  
  img = cv2.imread('objetos.jpg')  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

  # Umbralización Otsu  
  _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

  # Detectar contornos  
  contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  img_contours = img.copy()  
  cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)  

  # Clasificación por área  
  classified = img.copy()  
  small, medium, large = [], [], []  
  for contour in contours:  
      area = cv2.contourArea(contour)  
      if area < 500: small.append(contour)  
      elif area < 1000: medium.append(contour)  
      else: large.append(contour)  

  # Dibujar contornos clasificados  
  cv2.drawContours(classified, small, -1, (0, 255, 0), 2)  # Verde - Pequeño  
  cv2.drawContours(classified, medium, -1, (0, 0, 255), 2)  # Azul - Mediano  
  cv2.drawContours(classified, large, -1, (255, 0, 0), 2)  # Rojo - Grande  

  # Visualización y conteo  
  plt.figure(figsize=(6, 6))  
  plt.imshow(cv2.cvtColor(classified, cv2.COLOR_BGR2RGB))  
  plt.title(f'Clasificados: {len(small)} pequeños, {len(medium)} medianos, {len(large)} grandes')  
  plt.show()  

  # Guardar imagen  
  cv2.imwrite('classified.jpg', classified)

