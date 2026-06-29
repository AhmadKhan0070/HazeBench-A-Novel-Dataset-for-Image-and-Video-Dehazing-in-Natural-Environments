import os
import cv2
import random
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = r"dataset"   # === Dataset path====
NUM_IMAGES = 000  # === total no of images====

# =====================================================
# PSNR
# =====================================================

def psnr(img1, img2):
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    return 10 * np.log10((255.0 ** 2) / (mse + 1e-8))

# =====================================================
# SSIM
# =====================================================

def ssim(img1, img2):

    img1 = np.nan_to_num(img1).astype(np.float64)
    img2 = np.nan_to_num(img2).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu12

    numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)

    denominator = (
        (mu1_sq + mu2_sq + C1)
        * (sigma1_sq + sigma2_sq + C2)
    )

    ssim_map = numerator / (denominator + 1e-8)

    return float(np.mean(ssim_map))

# =====================================================
# DARK CHANNEL
# =====================================================

def dark_channel(img, size=15):

    min_img = np.min(img, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )

    return cv2.erode(min_img, kernel)

# =====================================================
# ATMOSPHERIC LIGHT
# =====================================================

def atmospheric_light(img, dark):

    h, w = dark.shape

    num_pixels = max(int(h * w * 0.001), 1)

    dark_vec = dark.reshape(-1)
    img_vec = img.reshape(-1, 3)

    indices = np.argsort(dark_vec)[-num_pixels:]

    A = np.mean(img_vec[indices], axis=0)

    A = np.maximum(A, 1)

    return A

# =====================================================
# TRANSMISSION
# =====================================================

def transmission(img, A, omega=0.95):

    norm = img / (A + 1e-8)

    t = dark_channel(norm)

    t = np.clip(t, 0, 1)

    return 1 - omega * t

# =====================================================
# RECOVERY
# =====================================================

def recover(img, t, A, t0=0.1):

    t = np.maximum(t, t0)

    J = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        J[:, :, c] = (
            (img[:, :, c] - A[c]) / t
            + A[c]
        )

    J = np.nan_to_num(
        J,
        nan=0.0,
        posinf=255,
        neginf=0
    )

    J = np.clip(J, 0, 255)

    return J.astype(np.uint8)

# =====================================================
# DCP DEHAZE
# =====================================================

def dcp_dehaze(img):

    img = img.astype(np.float32)

    dark = dark_channel(img)

    A = atmospheric_light(img, dark)

    t = transmission(img, A)

    result = recover(img, t, A)

    return result

# =====================================================
# MAIN
# =====================================================

def main():

    image_paths = []

    for file in os.listdir(DATASET_PATH):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_paths.append(
                os.path.join(DATASET_PATH, file)
            )

    if len(image_paths) == 0:
        print("No images found.")
        return

    sample_size = min(NUM_IMAGES, len(image_paths))

    random_images = random.sample(
        image_paths,
        sample_size
    )

    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0

    for i, path in enumerate(random_images, start=1):

        img = cv2.imread(path)

        if img is None:
            continue

        try:

            output = dcp_dehaze(img)

            if np.isnan(output).any():
                continue

            p = psnr(output, img)
            s = ssim(output, img)

            if np.isnan(p) or np.isnan(s):
                continue

            total_psnr += p
            total_ssim += s
            valid_count += 1

            if i % 50 == 0:
                print(
                    f"Processed {i}/{sample_size}"
                )

        except Exception as e:
            print(f"Skipped: {path}")
            print(e)

    if valid_count == 0:
        print("No valid images processed.")
        return

    mean_psnr = total_psnr / valid_count
    mean_ssim = total_ssim / valid_count

    print("\n==========================")
    print("DCP EVALUATION RESULTS")
    print("==========================")
    print(f"Images Processed : {valid_count}")
    print(f"Mean PSNR        : {mean_psnr:.4f}")
    print(f"Mean SSIM        : {mean_ssim:.4f}")
    print("==========================")

if __name__ == "__main__":
    main()