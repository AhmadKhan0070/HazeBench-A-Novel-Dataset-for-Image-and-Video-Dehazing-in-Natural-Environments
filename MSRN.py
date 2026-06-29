import os
import cv2
import random
import numpy as np

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = r"dataset"   # === Dataset path====
NUM_IMAGES = 000  # === total no of images====


# ==========================================
# PSNR
# ==========================================

def psnr(img1, img2):

    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    return 10 * np.log10((255.0 ** 2) / (mse + 1e-8))


# ==========================================
# SSIM
# ==========================================

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


# ==========================================
# SINGLE SCALE RETINEX
# ==========================================

def single_scale_retinex(img, sigma):

    img = np.maximum(img, 1.0)

    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    blur = np.maximum(blur, 1.0)

    retinex = np.log10(img) - np.log10(blur)

    return retinex


# ==========================================
# MULTI SCALE RETINEX
# ==========================================

def multi_scale_retinex(img, sigmas):

    retinex = np.zeros_like(img, dtype=np.float64)

    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)

    retinex = retinex / len(sigmas)

    return retinex


# ==========================================
# MSR ENHANCEMENT
# ==========================================

def msr_enhancement(img):

    img = img.astype(np.float64) + 1.0

    sigmas = [15, 80, 250]

    result = multi_scale_retinex(img, sigmas)

    result = np.nan_to_num(result)

    for channel in range(3):

        channel_data = result[:, :, channel]

        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        if max_val - min_val < 1e-8:
            continue

        result[:, :, channel] = (
            (channel_data - min_val)
            / (max_val - min_val)
        ) * 255

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


# ==========================================
# MAIN
# ==========================================

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
        print("No images found")
        return

    sample_size = min(NUM_IMAGES, len(image_paths))

    selected_images = random.sample(
        image_paths,
        sample_size
    )

    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0

    for idx, path in enumerate(selected_images, start=1):

        img = cv2.imread(path)

        if img is None:
            continue

        try:

            output = msr_enhancement(img)

            if np.isnan(output).any():
                continue

            p = psnr(output, img)
            s = ssim(output, img)

            if np.isnan(p) or np.isnan(s):
                continue

            total_psnr += p
            total_ssim += s
            valid_count += 1

            if idx % 50 == 0:
                print(
                    f"Processed {idx}/{sample_size}"
                )

        except Exception as e:
            print("Skipped:", path)
            print(e)

    if valid_count == 0:
        print("No valid images processed")
        return

    mean_psnr = total_psnr / valid_count
    mean_ssim = total_ssim / valid_count

    print("\n==========================")
    print("MSR RESULTS")
    print("==========================")
    print(f"Images Processed : {valid_count}")
    print(f"Mean PSNR        : {mean_psnr:.4f}")
    print(f"Mean SSIM        : {mean_ssim:.4f}")
    print("==========================")


if __name__ == "__main__":
    main()