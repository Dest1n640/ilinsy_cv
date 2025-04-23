import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np
from skimage.morphology import binary_erosion
from skimage.color import rgb2hsv
from skimage.morphology.footprints import rectangle


def analyze_image(binary):
    labeled = label(binary)
    regions = regionprops(labeled)

    total_shapes = len(regions)

    circle_hues = {}
    rectangle_hues = {}

    for region in regions:
        is_circle = region.eccentricity < 0.5

        coords = region.coords

        region_hues = hsv_image[coords[:, 0], coords[:, 1], 0]

        unique_hues, counts = np.unique(region_hues, return_counts=True)
        dominant_hue = unique_hues[np.argmax(counts)]

        dominant_hue_degrees = int(dominant_hue * 360)

        hue_bin = (dominant_hue_degrees // 30) * 30

        if is_circle:
            if hue_bin in circle_hues:
                circle_hues[hue_bin] += 1
            else:
                circle_hues[hue_bin] = 1
        else:
            if hue_bin in rectangle_hues:
                rectangle_hues[hue_bin] += 1
            else:
                rectangle_hues[hue_bin] = 1

    return total_shapes, circle_hues, rectangle_hues


def count_shapes_and_rectangles(binary_mask):
    labeled = label(binary_mask)
    regions = regionprops(labeled)
    total_shapes = 0
    rectangle_count = 0

    for region in regions:
        # Прямоугольник обычно имеет эксцентриситет, близкий к 1 (вытянутый)
        if np.isclose(region.extent, 1.0, atol=0.1):
            rectangle_count += 1
        total_shapes += 1

    diff = total_shapes - rectangle_count
    return total_shapes, rectangle_count, diff


image = plt.imread("./balls_and_rects.png")
hsv_image = rgb2hsv(image)
gray = image.mean(axis=2)
binary = gray > 0

binary = binary_erosion(binary, np.ones((3, 3)))

total, rectangles, diff = count_shapes_and_rectangles(binary)
diff_hues = {}
print(f"Всего фигур: {total}")
print(f"Прямоугольников: {rectangles}")
print(f"Кругов: {diff}")

total_shapes, circle_shape, rectangle_shape = analyze_image(binary)

print(f"Колисество цветов - {total_shapes}")
print(f"Количество оттенков круга - {len(circle_shape) - 1}")
for value in circle_shape:
    print(f"Отенок круга - {value}")

print(f"Количество оттенков приямоугольника - {len(rectangle_shape) - 1}")
for value in rectangle_shape:
    print(f"Отенок приямоугольника - {value}")

# plt.subplot(121)
plt.imshow(hsv_image)
# plt.subplot(122)
# plt.plot(sorted(colors), "o-")
plt.show()
