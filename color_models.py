import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils_ import (
    bgr_to_rgb,
    rgb_to_bgr,
    rgb_to_cmyk,
    cmyk_to_rgb,
    rgb_to_hsi,
    rgb_to_hsv,
)


def resize_image(image, scale=0.4):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main():
    IMAGE_PATH = "img/food.jpeg"
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("NO IMAGE PATHH")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (80, 150), 60, (255, 0, 0), -1)    
        cv2.circle(img, (150, 150), 60, (0, 255, 0), -1)   
        cv2.circle(img, (220, 150), 60, (0, 0, 255), -1)   

    rgb = bgr_to_rgb(img)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    print("CONVERT TO RGB")

    c, m, y, k = rgb_to_cmyk(rgb)
    print("CONVERT TO CMYK")

    h_hsi, s_hsi, i_hsi = rgb_to_hsi(rgb)
    hsi_img = cv2.merge([h_hsi, s_hsi, i_hsi])
    print("CONVERT TO HSI")

    h_hsv, s_hsv, v_hsv = rgb_to_hsv(rgb)
    hsv_img = cv2.merge([h_hsv, s_hsv, v_hsv])
    print("CONVERT TO HSV")

    rgb_reconstructed = cmyk_to_rgb(c, m, y, k)
    bgr_reconstructed = rgb_to_bgr(rgb_reconstructed)
    scale = 0.35 if img.shape[0] > 800 else 1.0
    rgb_grid = np.hstack([r, g, b])
    cmyk_grid = np.hstack([c, m, y, k])
    hsi_grid = np.hstack([h_hsi, s_hsi, i_hsi])
    hsv_grid = np.hstack([h_hsv, s_hsv, v_hsv])

    cv2.imshow("ORIGINAL", resize_image(img, scale))
    cv2.imshow("RGB", resize_image(rgb_grid, scale))
    cv2.imshow("CMYK", resize_image(cmyk_grid, scale))
    cv2.imshow("HSI", resize_image(hsi_grid, scale))
    cv2.imshow("HSV", resize_image(hsv_grid, scale))
    cv2.imshow("HSI MERGE", resize_image(hsi_img, scale))
    cv2.imshow("HSV MERGE", resize_image(hsv_img, scale))
    cv2.imshow("CMYK RECON", resize_image(bgr_reconstructed, scale))

    print("\n PRESS ANY BT")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
