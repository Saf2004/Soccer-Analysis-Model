import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np

frame_counter = 0
step_size = 100
cap = cv2.VideoCapture("video.mp4")
while True:

    ret, frame = cap.read()

    if not ret:

        break

    if frame_counter % step_size==0:
        cv2.imwrite(f"frame_{frame_counter}.jpg",frame)

    frame_counter += 1

#%%

def find_blobs(image,min_blob_size = 5):

    height, width = image.shape

    blob_coordinates = []

    for y in range(height):

        for x in range(width):

            if image[y, x] != 0:

                min_x, max_x, min_y, max_y = find_bounding_box(image, x, y)

                bbox_width = max_x - min_x + 1
                bbox_height = max_y - min_y + 1

                if bbox_width >= min_blob_size and bbox_height >= min_blob_size:

                    middle_x = (min_x + max_x) // 2
                    middle_y = (min_y + max_y) // 2

                    blob_coordinates.append((middle_x, middle_y))

    return blob_coordinates


def find_bounding_box(image, x, y):
    height, width = image.shape

    min_x = width - 1
    max_x = 0
    min_y = height - 1
    max_y = 0

    coordinates_array = np.array([(x, y)])

    while coordinates_array.size > 0:
        x, y = coordinates_array[-1]
        coordinates_array = coordinates_array[:-1]

        if image[y, x] != 0:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            image[y, x] = 0

            if x > 0:
                coordinates_array = np.append(coordinates_array, [(x - 1, y)], axis=0)
            if x < width - 1:
                coordinates_array = np.append(coordinates_array, [(x + 1, y)], axis=0)
            if y > 0:
                coordinates_array = np.append(coordinates_array, [(x, y - 1)], axis=0)
            if y < height - 1:
                coordinates_array = np.append(coordinates_array, [(x, y + 1)], axis=0)

    return min_x, max_x, min_y, max_y

#%%
img = cv2.imread('frame_500.jpg')


img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_cut = img_hsv[185:630,0:1600]
img_cut_original = img_rgb[185:630,0:1000]

lower_red = np.array([0,103,174])
upper_red = np.array([180,255,255])

img_red = cv2.inRange(img_cut,lower_red,upper_red)

plt.imshow(img_red)
plt.show()

blob_coordinates_red = find_blobs(img_red)

circle_color = (0, 0, 255)
circle_radius = 50

for coord in blob_coordinates_red:
    x, yn = coord
    y = yn + 180
    cv2.circle(img_rgb, (x, y), circle_radius, circle_color, 2)

plt.imshow(img_rgb)