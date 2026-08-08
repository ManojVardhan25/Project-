# Project-
# ==========================================
# Deforestation Change Detection - Google Colab
# ==========================================

# Install dependencies
!pip install -q opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

print("Upload BEFORE satellite image")
before_upload = files.upload()
before_path = list(before_upload.keys())[0]

print("Upload AFTER satellite image")
after_upload = files.upload()
after_path = list(after_upload.keys())[0]

# Read images
before = cv2.imread(before_path)
after = cv2.imread(after_path)

# Resize to same dimensions
after = cv2.resize(after, (before.shape[1], before.shape[0]))

# Convert to grayscale
gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray_before, gray_after)

# Threshold to create binary change mask
_, change_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Remove small noise
kernel = np.ones((5,5), np.uint8)
change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

# Calculate forest loss percentage
changed_pixels = np.count_nonzero(change_mask)
total_pixels = change_mask.size
loss_percent = (changed_pixels / total_pixels) * 100

# Prediction
prediction = "Deforestation Detected" if loss_percent > 2 else "No Significant Deforestation"

# Overlay changed regions in red
overlay = before.copy()
overlay[change_mask == 255] = [0, 0, 255]

# Save outputs
cv2.imwrite("change_mask.png", change_mask)
cv2.imwrite("overlay_result.png", overlay)

# Display results
plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.title("Before")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
plt.title("After")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(change_mask, cmap="gray")
plt.title("Change Mask")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Detected Changes")
plt.axis("off")

plt.show()

print("\n========== RESULT ==========")
print("Prediction       :", prediction)
print(f"Forest Loss      : {loss_percent:.2f}%")
print("Changed Pixels   :", changed_pixels)
print("Total Pixels     :", total_pixels)
print("============================")

# Download results
files.download("change_mask.png")
files.download("overlay_result.png")
