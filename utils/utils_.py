import cv2
import numpy as np


def bgr_to_rgb(image):
    return image[:, :, ::-1].copy()

def rgb_to_bgr(image):
    return image[:, :, ::-1].copy()

def rgb_to_cmyk(image):
    norm_img = image.astype(np.float64) / 255.0
    r = norm_img[:, :, 0]
    g = norm_img[:, :, 1]
    b = norm_img[:, :, 2]
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    c = np.zeros_like(r)
    m = np.zeros_like(g)
    y = np.zeros_like(b)
    mask = k < 1.0
    c[mask] = (1.0 - r[mask] - k[mask]) / (1.0 - k[mask])
    m[mask] = (1.0 - g[mask] - k[mask]) / (1.0 - k[mask])
    y[mask] = (1.0 - b[mask] - k[mask]) / (1.0 - k[mask])
    c_uint8 = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    m_uint8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    y_uint8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    k_uint8 = np.clip(k * 255.0, 0, 255).astype(np.uint8)
    return c_uint8, m_uint8, y_uint8, k_uint8

def cmyk_to_rgb(c, m, y, k):
    c_norm = c.astype(np.float64) / 255.0
    m_norm = m.astype(np.float64) / 255.0
    y_norm = y.astype(np.float64) / 255.0
    k_norm = k.astype(np.float64) / 255.0
    r = 255.0 * (1.0 - c_norm) * (1.0 - k_norm)
    g = 255.0 * (1.0 - m_norm) * (1.0 - k_norm)
    b = 255.0 * (1.0 - y_norm) * (1.0 - k_norm)
    r_uint8 = np.clip(r, 0, 255).astype(np.uint8)
    g_uint8 = np.clip(g, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(b, 0, 255).astype(np.uint8)
    return cv2.merge([r_uint8, g_uint8, b_uint8])

def rgb_to_hsi(image):
    norm_img = image.astype(np.float64) / 255.0
    r = norm_img[:, :, 0]
    g = norm_img[:, :, 1]
    b = norm_img[:, :, 2]
    eps = 1e-6
    i = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    s = 1.0 - (3.0 / (sum_rgb + eps)) * min_rgb
    s[sum_rgb == 0] = 0.0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    h = theta.copy()
    h[b > g] = 2.0 * np.pi - h[b > g]
    h_deg = np.degrees(h)
    h_uint8 = np.clip((h_deg / 360.0) * 255.0, 0, 255).astype(np.uint8)
    s_uint8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    i_uint8 = np.clip(i * 255.0, 0, 255).astype(np.uint8)
    return h_uint8, s_uint8, i_uint8

def rgb_to_hsv(image):
    norm_img = image.astype(np.float64) / 255.0
    r = norm_img[:, :, 0]
    g = norm_img[:, :, 1]
    b = norm_img[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    eps = 1e-6

    v = cmax

    s = np.zeros_like(v)
    mask = cmax > 0
    s[mask] = delta[mask] / cmax[mask]

    h = np.zeros_like(v)
    mask_r = (delta > 0) & (cmax == r)
    mask_g = (delta > 0) & (cmax == g)
    mask_b = (delta > 0) & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps)) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + eps)) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + eps)) + 4)

    h_uint8 = np.clip(h / 2.0, 0, 179).astype(np.uint8)
    s_uint8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    v_uint8 = np.clip(v * 255.0, 0, 255).astype(np.uint8)

    return h_uint8, s_uint8, v_uint8

def bgr_to_cmyk(image_bgr):
    rgb = bgr_to_rgb(image_bgr)
    return rgb_to_cmyk(rgb)


def bgr_to_hsi(image_bgr):
    rgb = bgr_to_rgb(image_bgr)
    return rgb_to_hsi(rgb)


def bgr_to_hsv(image_bgr):
    rgb = bgr_to_rgb(image_bgr)
    return rgb_to_hsv(rgb)
