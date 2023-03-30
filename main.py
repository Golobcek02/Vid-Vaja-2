import cv2
import numpy as np


def my_roberts(slika):
    roberts_kernel_x = np.array([[1, 0], [0, -1]])
    roberts_kernel_y = np.array([[0, 1], [-1, 0]])

    output_img = np.zeros_like(slika)
    for i in range(1, slika.shape[0] - 1):
        for j in range(1, slika.shape[1] - 1):
            gradient_x = np.sum(slika[i - 1:i + 1, j - 1:j + 1] * roberts_kernel_x)
            gradient_y = np.sum(slika[i - 1:i + 1, j - 1:j + 1] * roberts_kernel_y)
            magnitude = np.hypot(gradient_x, gradient_y)
            output_img[i, j] = magnitude

    slika_robov = ((output_img - output_img.min()) / (output_img.max() - output_img.min())) * 255
    slika_robov = cv2.convertScaleAbs(slika_robov)
    return slika_robov


def my_prewitt(slika):
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    izhodna_slika = np.zeros_like(slika, dtype=np.float32)
    for i in range(1, slika.shape[0] - 1):
        for j in range(1, slika.shape[1] - 1):
            gradient_x = np.sum(slika[i - 1:i + 2, j - 1:j + 2] * prewitt_kernel_x)
            gradient_y = np.sum(slika[i - 1:i + 2, j - 1:j + 2] * prewitt_kernel_y)
            magnituda = np.hypot(gradient_x, gradient_y)
            izhodna_slika[i, j] = magnituda
    slika_robov = ((izhodna_slika - izhodna_slika.min()) / (izhodna_slika.max() - izhodna_slika.min())) * 255
    slika_robov = cv2.convertScaleAbs(slika_robov)
    return slika_robov


def my_sobel(slika):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    output_img = np.zeros_like(slika, dtype=np.float32)
    for i in range(1, slika.shape[0] - 1):
        for j in range(1, slika.shape[1] - 1):
            gradient_x = np.sum(slika[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_x)
            gradient_y = np.sum(slika[i - 1:i + 2, j - 1:j + 2] * sobel_kernel_y)
            magnitude = np.hypot(gradient_x, gradient_y)
            output_img[i, j] = magnitude

    slika_robov = ((output_img - output_img.min()) / (output_img.max() - output_img.min())) * 255
    slika_robov = cv2.convertScaleAbs(slika_robov)
    return slika_robov


def canny(slika, sp_prag, zg_prag):
    # vaša implementacija
    slika_robov = 0
    return slika_robov


def spremeni_kontrast(slika, alfa, beta):
    pass


slika = cv2.imread(r'D:\FERI\4_FERI_NALOGE\Vid\Vaja_2_Git\Lena.png', cv2.IMREAD_GRAYSCALE)
slika = my_sobel(slika)
cv2.namedWindow("Slika")
slika2 = my_sobel(slika)
cv2.namedWindow("Slika2")

while True:
    cv2.imshow("Slika", slika)
    cv2.imshow("Slika2", slika2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
