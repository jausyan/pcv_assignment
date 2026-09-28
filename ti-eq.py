import cv2
import numpy as np

def image_negative(image):
    return 255 - image

def log_transform(image, c=None):
    img_float = image.astype(np.float64)
    if c is None:
        c = 255.0 / np.log(1.0 + np.max(img_float) + 1e-5)
    
    log_img = c * np.log(1.0 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def power_law_transform(image, gamma=1.0, c=1.0):
    norm_img = image.astype(np.float64) / 255.0
    gamma_img = c * (norm_img ** gamma)
    rescaled_img = gamma_img * 255.0
    return np.clip(rescaled_img, 0, 255).astype(np.uint8)

def calculate_histogram(channel):
    hist = np.zeros(256, dtype=np.int64)
    for pixel_value in channel.flat:
        hist[pixel_value] += 1
    return hist

def equalize_channel(channel):
    hist = calculate_histogram(channel)
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_m.min()
    cdf_max = cdf_m.max()
    
    if cdf_max == cdf_min:
        return channel.copy()
        
    cdf_normalized = (cdf_m - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
    equalized_channel = cdf_final[channel]
    return equalized_channel


def histogram_equalization(image):
    if len(image.shape) == 2:  
        return equalize_channel(image)
    elif len(image.shape) == 3:  
        b, g, r = cv2.split(image)
        eq_b = equalize_channel(b)
        eq_g = equalize_channel(g)
        eq_r = equalize_channel(r)
        return cv2.merge((eq_b, eq_g, eq_r))
    return image

if __name__ == "__main__":
    IMAGE_PATH = "img/food.jpeg"
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("NO IMG PATH")
        x = np.linspace(0, 255, 300, dtype=np.uint8)
        img = np.tile(x, (300, 1))

    neg_img = image_negative(img)
    log_img = log_transform(img)
    gamma_dark = power_law_transform(img, gamma=2.2)  # Darken
    gamma_bright = power_law_transform(img, gamma=0.4) # Brighten
    eq_img = histogram_equalization(img)
    cv2.imshow("ori", img)
    cv2.imshow("Image Negative", neg_img)
    cv2.imshow("Log Transform", log_img)
    cv2.imshow("Gamma (Gamma=0.4 - Bright)", gamma_bright)
    cv2.imshow("Gamma (Gamma=2.2 - Dark)", gamma_dark)
    cv2.imshow("Histogram Equalized", eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()