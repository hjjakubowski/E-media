import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt




def fft_display(image):
    img = cv.imread(image, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read the image at")

    if len(img.shape) == 2:

        img_fft = np.fft.fft2(img)
        shifted_fft = np.fft.fftshift(img_fft)
        shifted_fft_log_scale = np.log(np.abs(shifted_fft) + 1)

        plt.figure(figsize=(5, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original png image")
        plt.subplot(2, 1, 2)
        plt.imshow(shifted_fft_log_scale, cmap="gray")
        plt.title("Log Power Spectrum of the image")
        plt.show()
    else:
        img = img.copy()

        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGB)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        shifted_fft_log_scale = np.zeros_like(img, dtype=np.float32)

        for i in range(3):
            channel = img[:, :, i]
            channel_to_fft = np.fft.fft2(channel)
            shifted_fft_channel = np.fft.fftshift(channel_to_fft)
            shifted_fft_log_scale[:,:,i] = np.log(np.abs(shifted_fft_channel) + 1)

        max_val = np.max(shifted_fft_log_scale)
        if max_val > 0:
            shifted_fft_log_scale = shifted_fft_log_scale / max_val

        plt.figure(figsize=(5, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title("Original png image")
        plt.subplot(2, 1, 2)
        plt.imshow(shifted_fft_log_scale)
        plt.title("Log Power Spectrum of the image")
        plt.show()

