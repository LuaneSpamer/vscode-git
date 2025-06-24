import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
#from sklearn.linear_model import LinearRegression

image_paths = [
    "/path/to/ExposureTest_1.png",
    "/path/to/ExposureTest_2.png",
    "/path/to/ExposureTest_3.png",
    "/path/to/ExposureTest_4.png",
    "/path/to/ExposureTest_5.png",
    "/path/to/ExposureTest_6.png",
    "/path/to/ExposureTest_7.png",
    "/path/to/ExposureTest_8.png",
    "/path/to/ExposureTest_9.png",
    "/path/to/ExposureTest_10.png",
    "/path/to/ExposureTest_11.png",
    "/path/to/ExposureTest_12.png",
    "/path/to/ExposureTest_13.png",
    "/path/to/ExposureTest_14.png",
    "/path/to/ExposureTest_15.png",
    "/path/to/ExposureTest_16.png",
    "/path/to/ExposureTest_17.png",
    "/path/to/ExposureTest_18.png",
]

assert len(image_paths) % 2 == 0, "Image list must contain an even number of files (paired images)."

#exposure_times = np.array([50, 100, 250, 500, 1000, 2000, 5000], dtype=np.float32)  # in milliseconds
exposure_times = np.array([5, 10, 20, 40, 80, 160, 320, 640, 900], dtype=np.float32)  # in milliseconds

# Initialize lists to hold mean intensity and image intensity variance correscponding to exposure_times
mean_intensities = []
variances = []

for i in range(0, len(image_paths), 2):
    path1 = image_paths[i]
    path2 = image_paths[i + 1]

    # Load both images as float32 for precision
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # ***
    # Compute average intensity image (per-pixel)
    #avg_image = 0.5 * (img1 + img2)

    # Mean signal (It)
    #It = avg_image.mean()
    # ***

    # Total number of pixels
    Np = img1.size

    # Compute mean intensity (It)
    It = np.sum(img1 + img2) / (2 * Np)



    # Compute variance (Vt^2)
    diff_squared = (img1 - img2) ** 2
    Vt_squared = np.sum(diff_squared) / (2 * Np)

    # Store results
    mean_intensities.append(It)
    variances.append(Vt_squared)

    print(f"Pair {i//2 + 1}: It = {It:.2f}, Vt^2 = {Vt_squared:.4f}")

# Convert to arrays ??
mean_intensities = np.array(mean_intensities)
variances = np.array(variances)    

plt.plot(exposure_times, mean_intensities, 'bo-', label="Measured")

# # DETERMINE tSat

# # --- Step 1: Find bounding points ---
# threshold = 0.15  # 15% deviation

# i = 2  # Start at third point (need at least 2 for regression)
# while i < len(exposure_times):
#     # Fit linear regression up to the (i-1)th point
#     fit = np.polyfit(mean_intensities[:i], variances[:i], 1)
#     predicted = np.polyval(fit, mean_intensities[i])
#     measured = variances[i]
    
#     # Check deviation
#     deviation = abs(measured - predicted) / predicted

#     if deviation > threshold:
#         t_low = exposure_times[i-1]
#         t_high = exposure_times[i]
#         print(f"Bounding points found: t_low = {t_low}, t_high = {t_high}")
#         break
#     i += 1
# else:
#     raise ValueError("Linear range not exceeded in provided data.")

# # --- Step 2: Golden Section Search for tSat (max variance) ---
# def simulate_variance_at_t(t_query):
#     # Interpolate intensity and simulate predicted variance under linear model
#     # Here, we assume the variance follows a parabolic profile near saturation
#     # For testing, you might replace this with a function using real data
#     return np.interp(t_query, exposure_times, variances)

# res = minimize_scalar(
#     lambda t: -simulate_variance_at_t(t),  # maximize variance => minimize negative
#     method='golden',
#     bracket=(t_low, t_high),
#     tol=(0.001 / 100) * t_high**2
# )

# t_sat = res.x
# print(f"Estimated tSat = {t_sat:.6f} s")

# # --- Optional: Plotting for visualization ---
# plt.plot(mean_intensities, variances, 'bo-', label="Measured")
# plt.axvline(np.interp(t_sat, exposure_times, mean_intensities), color='red', linestyle='--', label=f'tSat ≈ {t_sat:.3f}s')
# plt.xlabel("Mean Intensity (It)")
# plt.ylabel("Variance (Vt^2)")
# plt.title("Variance vs. Intensity")
# plt.legend()
# plt.grid(True)
# plt.show()