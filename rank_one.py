import os
import cv2
import random
import numpy as np

# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = r"dataset"   # === Dataset path====
NUM_IMAGES = 000  # === total no of images====


# =====================================================
# PSNR (stable)
# =====================================================

def psnr(img1, img2):
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    return 10 * np.log10((255.0 ** 2) / (mse + 1e-8))

# =====================================================
# SSIM (stable version)
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

    return float(np.mean(numerator / (denominator + 1e-8)))

# =====================================================
# SAFE SVD (CRITICAL FIX)
# =====================================================

def safe_rank_one(channel):

    # convert to float
    channel = channel.astype(np.float64)

    # remove NaN/inf BEFORE SVD
    channel = np.nan_to_num(channel)

    try:
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)

        # keep only strongest component
        S = np.nan_to_num(S)

        S[1:] = 0

        result = np.dot(U, np.dot(np.diag(S), Vt))

        result = np.nan_to_num(result)

        return result

    except Exception:
        # fallback (no crash)
        return channel

# =====================================================
# RANK-ONE DEHAZING MODEL
# =====================================================

def rank_one_plus(img):

    img = img.astype(np.float32)

    output = np.zeros_like(img)

    for c in range(3):

        output[:, :, c] = safe_rank_one(img[:, :, c])

    output = np.nan_to_num(output)

    output = np.clip(output, 0, 255)

    return output.astype(np.uint8)

# =====================================================
# MAIN EVALUATION
# =====================================================

def main():

    images = [
        os.path.join(DATASET_PATH, f)
        for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) == 0:
        print("No images found")
        return

    sample_size = min(NUM_IMAGES, len(images))
    selected = random.sample(images, sample_size)

    total_psnr = 0.0
    total_ssim = 0.0
    valid = 0

    for i, path in enumerate(selected, 1):

        img = cv2.imread(path)

        if img is None:
            continue

        try:

            output = rank_one_plus(img)

            if np.isnan(output).any():
                continue

            p = psnr(output, img)
            s = ssim(output, img)

            if np.isnan(p) or np.isnan(s):
                continue

            total_psnr += p
            total_ssim += s
            valid += 1

            if i % 50 == 0:
                print(f"Processed {i}/{sample_size}")

        except Exception as e:
            print("Skipped:", path)

    if valid == 0:
        print("No valid images processed")
        return

    print("\n==========================")
    print("RANK-ONE PLUS RESULTS")
    print("==========================")
    print("Images:", valid)
    print("Mean PSNR:", total_psnr / valid)
    print("Mean SSIM:", total_ssim / valid)
    print("==========================")

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()