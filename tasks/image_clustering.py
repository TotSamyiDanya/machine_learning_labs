import cv2
import numpy as np
from matplotlib import pyplot as plt


def clusterize_image():
    image = cv2.imread('data/me.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    segment_image(image, 2, pixel_values, criteria)
    segment_image(image, 3, pixel_values, criteria)
    segment_image(image, 5, pixel_values, criteria)
    segment_image(image, 7, pixel_values, criteria)


def segment_image(image, k, pixel_values, criteria):
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    plt.imshow(segmented_image)
    plt.show()
