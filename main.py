import cv2
import numpy as np


def selected_edges(slika, edge):
    slika_robov = np.zeros((slika.shape[0], slika.shape[1], 3), np.uint8)
    slika = cv2.cvtColor(slika, cv2.COLOR_GRAY2BGR)
    slika_robov[:, :, 0] = edge  # Change the blue channel instead of the red channel
    slika_robov = cv2.addWeighted(slika, 0.4, slika_robov, 1.4, 3)
    cv2.imshow("Selected edges", slika_robov)


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
    selected_edges(slika, slika_robov)
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
    selected_edges(slika, slika_robov)
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
    selected_edges(slika, slika_robov)
    return slika_robov


def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, sp_prag, zg_prag)
    selected_edges(slika, slika_robov)
    return slika_robov


def spremeni_kontrast(slika, alfa, beta):
    slika = slika.astype(np.float32)
    slika_kontrast = alfa * slika + beta
    slika_kontrast = np.clip(slika_kontrast, 0, 255)
    slika_kontrast = slika_kontrast.astype(np.uint8)
    return slika_kontrast


slika = cv2.imread(r'D:\FERI\4_FERI_NALOGE\Vid\Vaja_2_Git\Lena.png', 0)
slika = spremeni_kontrast(slika, 1, 180)

funkcija = 1

while True:
    if funkcija == 1:
        cv2.imshow("Roberts", my_roberts(slika))
    elif funkcija == 2:
        cv2.imshow("Prewitt", my_prewitt(slika))
    elif funkcija == 3:
        cv2.imshow("Sobel", my_sobel(slika))
    elif funkcija == 4:
        cv2.imshow("Canny", canny(slika))
    else:
        break
    cv2.imshow("Kontrast", slika)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
