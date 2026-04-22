import cv2
import image_dehazer
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim


if __name__ == "__main__":


    # Path to the hazy images dataset
    dataset_path = 'classified_images/Urban_Areas'

    # Path to save dehazed images
    output_path = 'output_Existing/Urban'

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Lists to store PSNR values and image filenames
    psnr_values = []
    ssim_values = []
    image_filenames = []
    processing_times = []


    def calculate_psnr(original, dehazed):
      mse = np.mean((original - dehazed) ** 2)
      max_value = np.max(original)
      psnr = 20 * np.log10(max_value / np.sqrt(mse))
      return psnr

    # Iterate over the hazy images in the dataset
    for filename in os.listdir(dataset_path):
      if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load hazy image
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)

        # Start timer
        start_time = time.time()

        # Apply haze removal
        dehazed_image, haze_map = image_dehazer.remove_haze(image, showHazeTransmissionMap=False)		# Remove Haze

        # Save dehazed image
        output_image_path = os.path.join(output_path, filename)
        cv2.imwrite(output_image_path, dehazed_image)

        # Calculate PSNR value
        original_image = cv2.imread(image_path)
        psnr = calculate_psnr(original_image, dehazed_image)

        win_size = 3
        ssim_value, _ = ssim(original_image, dehazed_image,win_size = win_size,  full=True)
        ssim_values.append(ssim_value)


        # Store PSNR value and filename
        psnr_values.append(psnr)
        image_filenames.append(filename)

          # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)

        print(f"Dehazed image saved: {output_image_path}, PSNR: {psnr} , SSIM:{ssim_value}, Processing Time: {processing_time:.2f} seconds")

# Calculate mean PSNR value
mean_psnr = np.nanmean(psnr_values)

mean_ssim = np.mean(ssim_values)
mean_processing_time = np.mean(processing_times)

print("Mean PSNR:", mean_psnr)
print("Mean Processing Time:", mean_processing_time, "seconds")


# Plot the PSNR values
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(psnr_values) + 1),
    psnr_values,
    marker='o',
    linestyle='-',
    color='blue',
    markerfacecolor='blue',
    markersize=8,
    linewidth=2,
    label='PSNR Values'
)

# Add mean PSNR as a horizontal line
plt.axhline(y=mean_psnr, color='red', linestyle='--', linewidth=1.5, label=f'Mean PSNR: {mean_psnr:.2f}')

# Add labels, title, and grid
plt.xlabel('Image Pair', fontsize=12, weight='bold')
plt.ylabel('PSNR (dB)', fontsize=12, weight='bold')
plt.title('PSNR Values per Image Pair', fontsize=14, weight='bold')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=10, loc='lower right')

# Customize ticks
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Show the plot
plt.tight_layout()
plt.show()


# Plot the SSIM values
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(ssim_values) + 1),
    ssim_values,
    marker='o',
    linestyle='-',
    color='blue',
    markerfacecolor='blue',
    markersize=8,
    linewidth=2,
    label='SSIM Values'
)

# Add mean PSNR as a horizontal line
plt.axhline(y=mean_ssim, color='red', linestyle='--', linewidth=1.5, label=f'Mean SSIM: {mean_ssim:.2f}')

# Add labels, title, and grid
plt.xlabel('Image Pair', fontsize=12, weight='bold')
plt.ylabel('SSIM (dB)', fontsize=12, weight='bold')
plt.title('SSIM Values per Image Pair', fontsize=14, weight='bold')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=10, loc='lower right')

# Customize ticks
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

# Show the plot
plt.tight_layout()
plt.show()


print("Dehazing and PSNR calculation complete!")