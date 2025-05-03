import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import sobel, threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes


def find_pencil(image_region, min_size):
    image_height, image_width = image_region.image.shape
    center_y, center_x = image_region.centroid_local

    normalized_x = center_x / image_width
    normalized_y = center_y / image_height
    is_centered = (0.4 < normalized_x < 0.6) and (0.4 < normalized_y < 0.6)

    image_diagonal = (image_height**2 + image_width**2) ** 0.5
    is_size_valid = (image_diagonal > min_size / 2) and (image_diagonal < min_size)

    perimeter_ratio = image_region.perimeter / image_diagonal
    is_shape_valid = 2.5 < perimeter_ratio < 5.5

    is_elongated = (image_region.perimeter**2) / image_region.area > 33.3

    return is_centered and is_size_valid and is_shape_valid and is_elongated


count_all = 0

for i in range(1, 13):
    image = plt.imread(f"./images/img ({i}).jpg")
    gray_image = rgb2gray(image) * 255
    edges = sobel(gray_image)

    thresh = threshold_otsu(edges) / 2
    binary = edges >= thresh

    cleaned = binary_fill_holes(binary, structure=np.ones((3, 3)))

    labeled = label(cleaned)
    regions = regionprops(labeled)
    regions = sorted(regions, key=lambda item: item.perimeter)

    size = np.min(labeled.shape)
    count = sum(1 for region in regions if find_pencil(region, size))

    print(f"На {i} изображении: {count} карандашей")
    count_all += count

print(f"\nОбщее количество карандашей: {count_all}")
