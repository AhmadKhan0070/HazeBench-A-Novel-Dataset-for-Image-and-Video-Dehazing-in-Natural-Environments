import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Path to the hazy images dataset
dataset1_path = 'output_Existing/output urban areas'
dataset2_path = 'output_ownModel/Urban'

# Path to save dehazed images
output_path = 'final_output/urban'


# Lists to store PSNR values
psnr_values = []
ssim_values = []


# Iterate over the hazy images in the dataset
for filename in os.listdir(dataset1_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load hazy image
        image1_path = os.path.join(dataset1_path, filename)
        image1 = cv2.imread(image1_path)

        image2_path = os.path.join(dataset2_path, filename)
        image2 = cv2.imread(image2_path)


    # Make sure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Both images should have the same dimensions.")

    # Convert images to float32 for accurate pixel-wise minimum calculation
    image1 = np.float32(image1)
    image2 = np.float32(image2)

    # Perform min fusion
    fused_image = np.minimum(image1, image2)

    # Convert back to uint8 for visualization and saving
    fused_image = np.uint8(fused_image)

    # Save dehazed image
    output_image_path = os.path.join(output_path, filename + ".png")
    cv2.imwrite(output_image_path, fused_image)


    # Calculate PSNR
    mse = np.mean((image1 - fused_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    psnr_values.append(psnr)

    win_size = 3
    ssim_value, _ = ssim(image1, fused_image,win_size = win_size,  full=True ,   data_range=1.0)
    ssim_values.append(ssim_value)

    print(f"fusion image saved: {output_path}, PSNR: {psnr}")


# Calculate the mean PSNR
mean_psnr = np.nanmean(
    np.where(np.isinf(psnr_values), np.nan, psnr_values)
)
mean_ssim = np.mean(ssim_values)

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




