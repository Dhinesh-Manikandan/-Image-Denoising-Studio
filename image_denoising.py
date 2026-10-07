import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def add_salt_and_pepper_noise(image, amount=0.04):
    """
    Adds salt and pepper noise to an image.
    """
    noisy = np.copy(image)
    
    # Add Salt (white pixels)
    # amount * 0.5 means half salt, half pepper
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    # Add Pepper (black pixels)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy

def main():
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog to select an image
    print("Please select an image file from the dialog...")
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:
        print("No image selected. Exiting...")
        return
        
    print(f"Image selected: {file_path}")
    
    # 1. Read the original image
    # OpenCV reads in BGR format, we'll convert it to RGB for matplotlib
    original_bgr = cv2.imread(file_path)
    if original_bgr is None:
        print("Error reading the image.")
        return
        
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Add salt-and-pepper noise
    print("Adding salt-and-pepper noise...")
    noisy_rgb = add_salt_and_pepper_noise(original_rgb, amount=0.05)
    
    # 3. Apply 3x3 median filter to remove the noise
    print("Applying 3x3 median filter...")
    # cv2.medianBlur takes the image and the kernel size (must be an odd number)
    denoised_rgb = cv2.medianBlur(noisy_rgb, 3)
    
    # 4. Display the three images using Matplotlib
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_rgb)
    plt.title("Noisy Image (Salt & Pepper)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_rgb)
    plt.title("Denoised Image (Median Filter)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show() # This will pause execution until the user closes the plot window
    
    # 5. Save the noisy and denoised images
    # Convert back to BGR for saving with OpenCV
    noisy_bgr = cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the output files in a dedicated 'output_images' directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output_images")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    noisy_path = os.path.join(output_dir, "noisy_image.png")
    denoised_path = os.path.join(output_dir, "denoised_image.png")
    
    cv2.imwrite(noisy_path, noisy_bgr)
    cv2.imwrite(denoised_path, denoised_bgr)
    
    print(f"Saved noisy image to: {noisy_path}")
    print(f"Saved denoised image to: {denoised_path}")
    print("Processing complete!")

if __name__ == "__main__":
    main()
