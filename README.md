# EXP-12--PROJECT-Face-Detection-with-Haar-Cascades

### Develop by : konduru santhosh
### Reg no: 212225240074

## Aim

To write a Python program using OpenCV to perform the following image manipulations:
i) Extract ROI from an image.
ii) Perform face detection using Haar Cascades in static images.
iii) Perform eye detection in images.
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above
- OpenCV library (`opencv-python`)
- Matplotlib library (`matplotlib`)
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files

### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces
- Step 3: Draw white rectangles around detected faces with thickness 10
- Step 4: Return the processed image with rectangles

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes
- Step 3: Draw white rectangles around detected eyes with thickness 10
- Step 4: Return the processed image with rectangles

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`
- Step 2: Loop to continuously read frames from webcam
- Step 3: Apply `detect_face()` function on each frame
- Step 4: Display the video frame with rectangles around detected faces
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed
- Step 6: Release video capture and destroy all OpenCV windows

## Program:


```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image and convert the image into RGB
image = cv2.imread("C:/Users/acer/OneDrive/Desktop/YOLOv4_Webcam/peacock.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("on")
plt.show()

```
<img width="490" height="325" alt="image" src="https://github.com/user-attachments/assets/e0a01fb6-3e5a-4617-8d32-32627ea4cc34" />


## Segmented ROI

```
roi = image[100:420, 200:550]  # ROI coordinates (adjust as needed)

# Create a blank mask of the same size as the original image
mask = np.zeros_like(image)

# Place the ROI on the mask
mask[100:420, 200:550] = roi
# Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
segmented_roi = cv2.bitwise_and(image, mask)

# Step 6: Display the segmented ROI from the image
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)
plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()

```
<img width="460" height="322" alt="image" src="https://github.com/user-attachments/assets/a7038030-28d5-408b-aa34-b2d21f649a43" />


## II) Handwriting Detection in an Image


```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image and convert to RGB for displaying
image = cv2.imread("C:/Users/acer/OneDrive/Pictures/Screenshots/Screenshot 2026-08-28 220811.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Original Image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

```

<img width="433" height="377" alt="image" src="https://github.com/user-attachments/assets/8a566e23-1e28-402b-abf9-8ca3e40b22d7" />



```



# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Step 3: Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur (5x5 kernel)

# Step 5: Use Canny edge detector to find edges
edges = cv2.Canny(blurred_image, 50, 150)  # Detect edges using Canny (thresholds 50 and 150)

# Canny Edge Detection
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

```

<img width="463" height="380" alt="image" src="https://github.com/user-attachments/assets/b8184549-a0ec-4957-a4e3-bbdc1c8df992" />



```


# Step 6: Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETRTR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 7: Filter contours based on area and draw bounding boxes
result_image = image.copy()  # Create a copy of the original image to draw bounding boxes
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter out small areas
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box for the contour
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle

# Handwriting Detection Result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')


```

<img width="487" height="367" alt="image" src="https://github.com/user-attachments/assets/ce316356-ab61-46c4-be88-d70021a87922" />


## III) Object Detection with Labels in an Image using MobileNet-SSD



```


import cv2
import matplotlib.pyplot as plt

image_path = "C:/Users/acer/OneDrive/Desktop/YOLOv4_Webcam/car.jpg"

image = cv2.imread(image_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = (30, 40, 30)
upper = (90, 255, 255)

mask = cv2.inRange(hsv, lower, upper)

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest)

cv2.rectangle(
    image,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    3
)

cv2.putText(
    image,
    "car",
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(image)
plt.axis("off")
plt.show()


```


<img width="597" height="380" alt="image" src="https://github.com/user-attachments/assets/9b299b20-267e-4e2b-a93f-99e28bf56ad6" />










































## Result

Thus, the Python program using OpenCV was successfully implemented to extract ROI, detect faces and eyes using Haar Cascades in static images, and detect faces with labels in real-time webcam video.
