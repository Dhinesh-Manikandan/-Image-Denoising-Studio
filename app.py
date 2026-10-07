import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from streamlit_paste_button import paste_image_button

def add_salt_and_pepper_noise(image, amount=0.04):
    """Adds salt and pepper noise to an image."""
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1 if i > 1 else 0, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1 if i > 1 else 0, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_gaussian_noise(image, mean=0, std=25):
    """Adds Gaussian noise to an image."""
    gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), gaussian)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_denoising(image, algo, filter_size, h_param):
    if algo == "Standard Median":
        return cv2.medianBlur(image, filter_size)
    elif algo == "Non-Local Means (NLM)":
        # State of the art classic algorithm for Gaussian noise
        return cv2.fastNlMeansDenoisingColored(image, None, h_param, h_param, 7, 21)
    elif algo == "Bilateral Filter":
        # Excellent for edge-preserving smoothing
        return cv2.bilateralFilter(image, filter_size, sigmaColor=75, sigmaSpace=75)
    return image

def main():
    st.set_page_config(page_title="Advanced Denoising Studio", layout="wide", page_icon="✨")
    
    st.title("✨ Advanced Image Denoising Studio")
    st.markdown("Upload or paste an image to test state-of-the-art classic denoising algorithms like Non-Local Means (NLM) and Bilateral Filtering.")
    
    st.sidebar.header("🛠️ Settings")
    
    noise_type = st.sidebar.selectbox("Noise Type", ["Salt & Pepper", "Gaussian"])
    if noise_type == "Salt & Pepper":
        noise_amount = st.sidebar.slider("Noise Intensity", 0.01, 0.20, 0.05, 0.01)
    else:
        noise_amount = st.sidebar.slider("Gaussian Std Dev", 5, 100, 25, 5)
        
    algo = st.sidebar.selectbox("Denoising Algorithm", [
        "Non-Local Means (NLM)", 
        "Bilateral Filter",
        "Standard Median"
    ])
    
    filter_size = 5
    h_param = 10
    
    if algo == "Standard Median" or algo == "Bilateral Filter":
        filter_size = st.sidebar.slider("Kernel / Filter Size", 3, 15, 5, 2)
    elif algo == "Non-Local Means (NLM)":
        h_param = st.sidebar.slider("Filter Strength (h)", 1.0, 30.0, 10.0, 1.0)
    
    uploaded_file = st.file_uploader("Drag and drop your image here", type=["jpg", "jpeg", "png"])
    
    st.markdown("Or paste an image directly from your clipboard:")
    paste_result = paste_image_button(
        label="📋 Paste Image",
        background_color="#FF4B4B",
        hover_background_color="#FF6B6B",
        errors="ignore"
    )
    
    original_rgb = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_bgr = cv2.imdecode(file_bytes, 1)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    elif paste_result.image_data is not None:
        original_rgb = np.array(paste_result.image_data)
        if len(original_rgb.shape) == 3 and original_rgb.shape[2] == 4:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_RGBA2RGB)
            
    if original_rgb is not None:
        st.divider()
        
        # Add Noise
        if noise_type == "Salt & Pepper":
            noisy_rgb = add_salt_and_pepper_noise(original_rgb, amount=noise_amount)
        else:
            noisy_rgb = add_gaussian_noise(original_rgb, std=noise_amount)
        
        # Denoise
        denoised_rgb = apply_denoising(noisy_rgb, algo, filter_size, h_param)
        
        # Save to output_images folder
        os.makedirs("output_images", exist_ok=True)
        cv2.imwrite(os.path.join("output_images", "noisy_image.png"), cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join("output_images", "denoised_image.png"), cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR))
        
        # Display images in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Original Image")
            st.image(original_rgb)
            
        with col2:
            st.subheader("2. Noisy Image")
            st.image(noisy_rgb)
            
            # Download button for Noisy
            noisy_pil = Image.fromarray(noisy_rgb)
            buf = io.BytesIO()
            noisy_pil.save(buf, format="PNG")
            st.download_button(label="📥 Download Noisy", data=buf.getvalue(), file_name="noisy_image.png", mime="image/png")
            
        with col3:
            st.subheader("3. Denoised Image")
            st.image(denoised_rgb)
            
            # Download button for Denoised
            denoised_pil = Image.fromarray(denoised_rgb)
            buf2 = io.BytesIO()
            denoised_pil.save(buf2, format="PNG")
            st.download_button(label="📥 Download Denoised", data=buf2.getvalue(), file_name="denoised_image.png", mime="image/png")

if __name__ == "__main__":
    main()
