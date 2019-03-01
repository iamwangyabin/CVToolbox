from PIL import Image
import numpy as np
import skimage.transform as st
import matplotlib.pyplot as plt
import Detector.HarrisCorner as harris

# 构建图片(二值图)
im = np.array(Image.open('1.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim, 6)

x,y=im.shape
image = np.zeros((x, y))
for i in filtered_coords:
    image[i[0],i[1]]=255

h, theta, d = st.hough_line(image)

fig, (ax0,ax2) = plt.subplots(1, 2, figsize=(8, 6))
plt.tight_layout()

ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()

ax2.imshow(image, plt.cm.gray)
row1, col1 = image.shape
for _, angle, dist in zip(*st.hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
    ax2.plot((0, col1), (y0, y1), '-r')
ax2.axis((0, col1, row1, 0))
ax2.set_title('Detected lines')
ax2.set_axis_off()


