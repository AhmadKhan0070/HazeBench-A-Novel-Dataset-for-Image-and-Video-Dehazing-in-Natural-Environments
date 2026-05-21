import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim


def dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percentage):
    flat_dark_channel = dark_channel.flatten()
    num_pixels = flat_dark_channel.size
    num_top_pixels = int(num_pixels * top_percentage / 100)
    indices = np.argpartition(flat_dark_channel, -num_top_pixels)[-num_top_pixels:]
    top_pixels = image.reshape(num_pixels, 3)[indices]
    atmospheric_light = np.max(top_pixels, axis=0)
    return atmospheric_light

def estimate_transmission(image, atmospheric_light, omega, window_size):
    normalized_image = image.astype(np.float32) / atmospheric_light.astype(np.float32)
    transmission = 1 - omega * dark_channel(normalized_image, window_size)
    return transmission

def dehaze(image, transmission, atmospheric_light, t0, omega):
    transmission_clamped = np.clip(transmission, t0, 1)
    restored_image = np.empty_like(image)
    for channel in range(3):
        restored_channel = (image[:, :, channel].astype(np.float32) - atmospheric_light[channel]) / transmission_clamped + atmospheric_light[channel]
        restored_image[:, :, channel] = np.clip(restored_channel, 0, 255)
    restored_image = restored_image.astype(np.uint8)
    return restored_image

def haze_removal(image, window_size=15, top_percentage=0.1, omega=0.95, t0=0.5):
    dark_channel_map = dark_channel(image, window_size)
    atmospheric_light = estimate_atmospheric_light(image, dark_channel_map, top_percentage)
    transmission = estimate_transmission(image, atmospheric_light, omega, window_size)
    restored_image = dehaze(image, transmission, atmospheric_light, t0, omega)
    return restored_image

def calculate_psnr(original, dehazed):
    mse = np.mean((original - dehazed) ** 2)
    max_value = np.max(original)
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr

# Path to the hazy images dataset
dataset_path = 'classified_images/Urban_Areas'

# Path to save dehazed images
output_path = 'output_ownModel/Urban'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Lists to store PSNR values and image filenames
psnr_values = []
ssim_values = []
image_filenames = []
processing_times = []

# Iterate over the hazy images in the dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load hazy image
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)

        # Start timer
        start_time = time.time()

        # Apply haze removal
        dehazed_image = haze_removal(image)

        # Save dehazed image
        output_image_path = os.path.join(output_path, filename)
        cv2.imwrite(output_image_path, dehazed_image)

        # Calculate PSNR value
        original_image = cv2.imread(image_path)
        psnr = calculate_psnr(original_image, dehazed_image)

        # Store PSNR value and filename
        psnr_values.append(psnr)
        image_filenames.append(filename)

        win_size = 3
        ssim_value, _ = ssim(original_image, dehazed_image,win_size = win_size,  full=True)
        ssim_values.append(ssim_value)

          # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)

        print(f"Dehazed image saved: {output_image_path}, PSNR: {psnr} , SSIM:{ssim_value}, Processing Time: {processing_time:.2f} seconds")




print("Dehazing and PSNR calculation complete!")

# Calculate mean PSNR value
mean_psnr = np.nanmean(
    np.where(np.isinf(psnr_values), np.nan, psnr_values)
)
mean_ssim = np.nanmean(ssim_values)
mean_processing_time = np.mean(processing_times)

# Plot the PSNR values
plt.figure()
plt.plot(range(1, len(psnr_values) + 1), psnr_values, marker='o', linestyle='-')
plt.xlabel('Image Pair')
plt.ylabel('PSNR')
plt.title(f'Mean PSNR: {mean_psnr:.2f}')
plt.grid(True)
plt.show()

# Plot the SSIM values
plt.figure()
plt.plot(range(1, len(ssim_values) + 1), ssim_values, marker='o', linestyle='-')
plt.xlabel('Image Pair')
plt.ylabel('SSIM')
plt.title(f'Mean SSIM: {mean_ssim:.2f}')
plt.grid(True)
plt.show()

print("Mean PSNR:", mean_psnr)
print("Mean SSIM:", mean_ssim)
print("Mean Processing Time:", mean_processing_time, "seconds")

