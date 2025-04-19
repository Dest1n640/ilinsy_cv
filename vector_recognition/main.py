from matplotlib.image import BboxImage
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import perimeter, regionprops, label


def extractor(region):
    img = region.image
    h, w = img.shape
    size = h * w

    norm_area = region.area / size
    y_center, x_center = region.centroid_local
    x_center /= w
    y_center /= h

    norm_perimeter = region.perimeter / size
    eccentricity = region.eccentricity
    defect = 1 - region.euler_number
    compactness = region.solidity

    column_fill = np.sum(np.all(img, axis=0))
    row_fill = np.sum(np.all(img, axis=1))

    aspect_ratio = w / h

    y_mid = int(y_center * h)
    x_mid = int(x_center * w)
    row_transitions = np.count_nonzero(img[y_mid, :-1] != img[y_mid, 1:])
    col_transitions = np.count_nonzero(img[:-1, x_mid] != img[1:, x_mid])

    upper = img[: h // 2, :]
    lower = img[-(h // 2) :, :]
    left = img[:, : w // 2]
    right = img[:, -(w // 2) :]

    horizontal_symmetry = np.sum(upper == np.flipud(lower)) / size
    vertical_symmetry = np.sum(left == np.fliplr(right)) / size

    result = np.array(
        [
            norm_area,
            y_center,
            x_center,
            norm_perimeter,
            eccentricity,
            defect,
            compactness,
            aspect_ratio,
            row_transitions,
            col_transitions,
            vertical_symmetry,
            horizontal_symmetry,
        ]
    )

    return result


def norm_l1(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5


def classificator(v, templates):
    result = "_"
    min_dist = 10**16
    for key in templates:
        d = norm_l1(v, templates[key])
        if d < min_dist:
            result = key
            min_dist = d
    return result


alphabet = plt.imread("./alphabet.png")[:, :, :-1]

gray = alphabet.mean(axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)
print(len(regions))

symbols = plt.imread("./alphabet-small.png")[:, :, :-1]
gray = symbols.mean(axis=2)
binary = gray < 1
slabeled = label(binary)
sregions = regionprops(slabeled)
print(len(regions))

templates = {
    "A": extractor(sregions[2]),
    "B": extractor(sregions[3]),
    "8": extractor(sregions[0]),
    "0": extractor(sregions[1]),
    "1": extractor(sregions[4]),
    "W": extractor(sregions[5]),
    "X": extractor(sregions[6]),
    "*": extractor(sregions[7]),
    "-": extractor(sregions[9]),
    "/": extractor(sregions[8]),
}


print(templates)
# for i, region in enumerate(sregions):
#    v = extractor(region)
#    plt.subplot(2, 5, i + 1)
#    plt.title(classificator(v, templates))
#    plt.imshow(region.image)

result = {}
for region in regions:
    v = extractor(region)
    symbols = classificator(v, templates)
    result[symbols] = result.get(symbols, 0) + 1

print(result)
plt.show()
