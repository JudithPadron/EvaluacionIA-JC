import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Desactivar optimizaciones oneDNN (opcional si no afectan)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Rutas de imágenes y etiquetas
image_paths = [
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Uno.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Dos.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Tres.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Cuatro.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Cinco.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Seis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Siete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Ocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Nueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Diez.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Once.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Doce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Trece.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Catorce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Quince.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Dieciseis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Diecisiete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Dieciocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Diecinueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Mexico-Veinte.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Uno.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Dos.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Tres.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Cuatro.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Cinco.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Seis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Siete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Ocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Nueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Diez.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Once.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Doce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Trece.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Catorce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Quince.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Dieciseis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Diecisiete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Dieciocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Diecinueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/India-Veinte.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Uno.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Dos.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Tres.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Cuatro.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Cinco.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Seis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Siete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Ocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Nueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Diez.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Once.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Doce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Trece.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Catorce.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Quince.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Dieciseis.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Diecisiete.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Dieciocho.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Diecinueve.jpg',
    'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Africa-Veinte.jpg'
]

labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 0-Mexicanos, 1-Indios, 2-Africanos

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(64, 64)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Función para predecir el origen cultural
def predict_pattern(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Etiquetas de las clases
labels = ["Mexicano", "Indio", "Africano"]

# Cargar modelo
model = tf.keras.models.load_model('pattern_classifier_model.h5')

# Ruta de la imagen estática
static_image_path = 'C:/Users/JUDITHDELCARMENPADRO/Desktop/Imagenes/Culturas.jpg'

# Función para clasificar y enmarcar cada región de una imagen estática
def classify_and_draw_boxes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:  # Filtrar contornos pequeños
            roi = image[y:y+h, x:x+w] 
            if roi.size > 0:
                class_idx = predict_pattern(roi, model)
                label = labels[class_idx]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Clasificacion de Vestimentas", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Clasificar y mostrar la imagen estática
classify_and_draw_boxes(static_image_path)

# Función para clasificar imágenes en tiempo real desde la cámara web
def classify_realtime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:  # Filtrar contornos pequeños
            roi = frame[y:y+h, x:x+w]  
            if roi.size > 0:
                class_idx = predict_pattern(roi, model)
                label = labels[class_idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Captura de video desde la cámara web
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Presiona 'p' para capturar y clasificar desde la cámara.")
print("Presiona 's' para clasificar la imagen cargada desde el código.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el video.")
        break

    # Mostrar la imagen en vivo de la cámara
    cv2.imshow("Cámara en Vivo - Presiona 'p' para predecir, 's' para imagen estatica", frame)

    # Leer el teclado
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Presionar 'p' para capturar y clasificar
        processed_frame = classify_realtime(frame.copy())
        cv2.imshow("Clasificacion de Vestimentas (Cámara)", processed_frame)
        cv2.waitKey(0)  # Esperar hasta que se cierre la ventana de predicción
        cv2.destroyWindow("Clasificacion de Vestimentas (Cámara)")
    elif key == ord('s'):  # Presionars 's' para clasificar la imagen cargada
        classify_static_image(static_image_path)
    elif key == ord('q'):  # Presionar 'q' para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
