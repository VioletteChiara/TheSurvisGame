import numpy as np
import cv2
def do_pie(values,colors, labels, color_fond, width):

    # Create a blank white image
    img = np.ones((200, width, 3), dtype=np.uint8) * 255
    img[:] = color_fond

    # Center and radius of the pie
    center = (int(width/2), 100)
    radius = 100

    # Draw the pie slices
    start_angle = 0
    total = sum(values)

    for value, color, label in zip(values, colors, labels):
        end_angle = start_angle + (value / total) * 360
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, -1)
        start_angle = end_angle

    # Optional: Draw a circle outline
    cv2.circle(img, center, radius, (0, 0, 0), 2)

    # Show image
    return(img)