import numpy as np
import cv2 as cv

img = cv.imread('dadu4.jpg')

assert img is not None, "file could not be read, check with os.path.exists()"

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Konversi citra marker ke tipe data CV_8UC1
markers = np.uint8(markers)

# Membuat citra kosong untuk menggambar kontur
contour_img = np.zeros_like(img, dtype=np.uint8)

# Menemukan kontur pada citra hasil segmentasi
contours, _ = cv.findContours(markers.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Menggambar dan mengisi kontur dengan warna tertentu (contoh: hijau)
cv.drawContours(contour_img, contours, -1, (0, 255, 0), thickness=cv.FILLED)

# Menambahkan hasilnya ke citra asli
result_img = cv.addWeighted(img, 1, contour_img, 0.5, 0)

cv.imshow("Hasil", result_img)
cv.imshow("Citra Awal", img)
cv.waitKey(0)
cv.destroyAllWindows()
