import cv2
import numpy as np

def pad_image(image, pad_height, pad_width, mode="reflect"):
    h, w = image.shape
    padded = np.zeros((h + 2 * pad_height, w + 2 * pad_width), dtype=np.float64)
    padded[pad_height : pad_height + h, pad_width : pad_width + w] = image

    if mode == "zero":
        return padded

    for i in range(pad_height):
        padded[i, pad_width : pad_width + w] = image[pad_height - i - 1, :]
        padded[h + pad_height + i, pad_width : pad_width + w] = image[h - 1 - i, :]

    for j in range(pad_width):
        padded[:, j] = padded[:, pad_width * 2 - j - 1]
        padded[:, w + pad_width + j] = padded[:, w + pad_width - 1 - j]

    return padded


def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    kernel_flipped = np.flipud(np.fliplr(kernel))
    padded_img = pad_image(image.astype(np.float64), pad_h, pad_w, mode="reflect")
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded_img[i : i + k_h, j : j + k_w]
            output[i, j] = np.sum(region * kernel_flipped)
    return output

def mean_filter(channel, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (
        kernel_size * kernel_size
    )
    result = convolve2d(channel, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_filter(channel, kernel_size=3, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)  
    result = convolve2d(channel, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)

def laplacian_filter(channel, scale_sharpen=1.0):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    laplacian = convolve2d(channel, kernel)
    sharpened = channel.astype(np.float64) - (scale_sharpen * laplacian)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def median_filter(channel, kernel_size=3):
    pad = kernel_size // 2
    padded_img = pad_image(channel.astype(np.float64), pad, pad, mode="reflect")
    h, w = channel.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            region = padded_img[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = np.median(region)
    return output

def apply_spatial_filter(image, filter_func, **kwargs):
    if len(image.shape) == 2:
        return filter_func(image, **kwargs)
    elif len(image.shape) == 3:
        b, g, r = cv2.split(image)
        fb = filter_func(b, **kwargs)
        fg = filter_func(g, **kwargs)
        fr = filter_func(r, **kwargs)
        return cv2.merge((fb, fg, fr))
    return image

if __name__ == "__main__":
    IMAGE_PATH = "img/food.jpeg"
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(
            "NO IMG"
        )
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), 255, -1)
        cv2.circle(img, (150, 150), 50, 0, -1)

    mean_img = apply_spatial_filter(img, mean_filter, kernel_size=5)
    gauss_img = apply_spatial_filter(img, gaussian_filter, kernel_size=5, sigma=1.5)
    med_img = apply_spatial_filter(img, median_filter, kernel_size=5)
    sharp_img = apply_spatial_filter(img, laplacian_filter, scale_sharpen=0.5)

    cv2.imshow("1. Real Image", img)
    cv2.imshow("2. Mean Filter (5x5)", mean_img)
    cv2.imshow("3. Gaussian Filter (5x5, sigma=1.5)", gauss_img)
    cv2.imshow("4. Median Filter (5x5)", med_img)
    cv2.imshow("5. Laplacian Sharpened", sharp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()