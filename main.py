import cv2
import numpy as np

def my_roberts(slika):
    #vaša implementacija
    slika_robov=0
    return slika_robov

def my_prewitt(slika):
    #vaša implementacija
    slika_robov=0
    return slika_robov

def my_sobel(slika):
    #vaša implementacija
    slika_robov=0
    return slika_robov

def canny(slika, sp_prag, zg_prag):
    #vaša implementacija
    slika_robov=0
    return slika_robov

def spremeni_kontrast(slika, alfa, beta):
    pass

img = cv2.imread(r'D:\FERI\4_FERI_NALOGE\Vid\Vaja_2_Git\Temp.jpg', cv2.IMREAD_GRAYSCALE)
my_roberts(img)
cv2.namedWindow("Slika")

while True:
    cv2.imshow("Slika", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
