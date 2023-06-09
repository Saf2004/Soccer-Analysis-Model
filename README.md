# Soccer Analysis Model

This model is designed to perform basic analysis on soccer video footage. It includes several functions for processing frames and extracting relevant information from the video.

## Prerequisites

Make sure you have the following dependencies installed:

- `cv2` (OpenCV)
- `matplotlib`
- `numpy`

## Usage

### Frame Extraction

To extract frames from a video, you can use the following code snippet:

```python
import cv2

frame_counter = 0
step_size = 100
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_counter % step_size == 0:
        cv2.imwrite(f"frame_{frame_counter}.jpg", frame)

    frame_counter += 1
```

Replace `"video.mp4"` with the path to your video file. This code will save frames at regular intervals defined by `step_size` (in this example, every 100 frames).

### Player Detection

The `find_blobs` function can be used to detect blobs (connected components) in a binary image. Here's an example usage:

```python
import numpy as np

def find_blobs(image, min_blob_size=5):
    # Function implementation...

image = cv2.imread('frame_500.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define lower and upper thresholds for blob detection
lower_red = np.array([0, 103, 174])
upper_red = np.array([180, 255, 255])

# Create binary image based on thresholding
img_red = cv2.inRange(img_hsv, lower_red, upper_red)

# Find blob coordinates
blob_coordinates_red = find_blobs(img_red)

# Visualize the blobs on the original image
circle_color = (0, 0, 255)
circle_radius = 50
for coord in blob_coordinates_red:
    x, yn = coord
    y = yn + 180
    cv2.circle(img_rgb, (x, y), circle_radius, circle_color, 2)

# Display the image with detected blobs
plt.imshow(img_rgb)
```

### Sub-Grid Analysis

The model includes functionality for analyzing a specific sub-grid within an image. Here's an example usage:

```python
def analyze_sub_grid(img_cut_original, blob_coordinates_blue, blob_coordinates_red):
    # Function implementation...

img = cv2.imread('frame_500.jpg')

# Perform necessary preprocessing on the image

grid_size = (3, 4)
row = 1
col = 2

# Extract the specified sub-grid from the original image
sub_grid = img_cut_original[start_y:end_y, start_x:end_x]

# Analyze the sub-grid to count blue and red team members
num_labels_blue, num_labels_red = analyze_sub_grid(sub_grid, blob_coordinates_blue, blob_coordinates_red)

# Display the sub-grid image
plt.imshow(sub_grid)
plt.show()

print("Number of blue team members in the sub-grid:", num_labels_blue)
print("Number of red team members in the sub-grid:", num_labels_red)
```

Make sure to replace the relevant variables (`img_cut_original`, `blob_coordinates_blue`, `blob_coordinates_red`, etc.) with your own data.

### Region Labeling

The model includes a function for detecting ellipses in an image. Here's an example usage:

```python
def find_longest_vertical_line(region):
    # Function implementation...

img = cv2.imread('frame_200.jpg')

# Perform necessary preprocessing on the image

lower = np.array([0, 0, 168])
upper = np.array([172, 255,

 255])

region = cv2.inRange(img_cut, lower, upper)

# Find the coordinates of the longest vertical line in the region
coord = find_longest_vertical_line(region)
x, yn = coord
y = yn + 150

# Draw an ellipse at the detected coordinates
image = cv2.ellipse(img_rgb, (x, y), (300, 80), 0, 0, 360, (0, 0, 255), 5)

# Display the image with the detected ellipse
plt.imshow(img_rgb)
```

Again, make sure to replace the relevant variables (`img`, `img_cut`, `img_rgb`, etc.) with your own data.

## Conclusion

This soccer analysis model provides basic functionalities for frame extraction, blob detection, sub-grid analysis, and ellipse detection in soccer video footage. Feel free to customize and enhance the model based on your specific needs and requirements. Enjoy analyzing your soccer videos!
