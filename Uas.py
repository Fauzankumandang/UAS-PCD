import cv2
import numpy as np

# Baca citra
img = cv2.imread("dadu.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

# Invers hasil thresholding agar latar belakang menjadi hitam (0) dan objek putih (255)
thresholded = cv2.bitwise_not(thresholded)

# Tampilkan hasil segmentasi
cv2.imshow("Hasil Segmentasi", thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the dimensions of the image
height, width, _ = img.shape

dataAVG = img.reshape((height * width), 3)

# Tentukan rentang warna dadu dalam bentuk HSV atau RGB
lower_color = np.array([0, 0, 0])  # Atur nilai sesuai warna dadu
upper_color = np.array([255, 50, 255])  # Atur nilai sesuai warna dadu

# Ubah citra ke ruang warna HSV jika diperlukan
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Buat mask untuk segmentasi warna dadu
mask = cv2.inRange(hsv, lower_color, upper_color)

# Aplikasikan mask ke citra asli
result = cv2.bitwise_and(img, img, mask=mask)

# Operasi morfologi (contoh: opening)
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Temukan kontur Dadu
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Gambar kotak pembatas di sekitar dadu
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Tampilkan citra asli dan hasil segmentasi
cv2.imshow("Citra Asli", img)
cv2.imshow("Hasil Segmentasi", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


def distance(abg1, avg2):
    point1 = np.array(avg1)
    point2 = np.array(avg2)

    dist = np.linalg.norm(point1 - point2)
    return dist


def countAvg(im):
    (x, y) = im.shape
    im = im.reshape(-1)
    px = [a for a in im if a > 0]
    return np.average(px)


def avgchannels(im, mask):
    imgR = im[:, :, 0]
    imgG = im[:, :, 1]
    imgB = im[:, :, 2]

    segR = cv2.bitwise_and(imgR, mask)
    segG = cv2.bitwise_and(imgG, mask)
    segB = cv2.bitwise_and(imgB, mask)

    avgR = countAvg(segR)
    avgG = countAvg(segG)
    avgB = countAvg(segB)
    return [avgR, avgG, avgB]
