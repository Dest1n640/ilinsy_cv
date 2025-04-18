import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.color import rgb2gray
from pathlib import Path


def count_holes(region):
    image = np.pad(region.image, pad_width=1, mode="constant", constant_values=0)
    inverted = ~image
    labeled = label(inverted)
    return np.max(labeled) - 1


def count_vlines(region):
    return np.all(region.image, axis=0).sum()


def is_P(region):
    # Проверка: есть дырка в верхней части и нет во всей нижней
    top_half = region.image[: region.image.shape[0] // 2, :]
    bottom_half = region.image[region.image.shape[0] // 2 :, :]
    holes_top = np.max(label(~top_half)) - 1
    holes_bot = np.max(label(~bottom_half)) - 1
    return holes_top >= 1 and holes_bot == 0


def recognize(region):
    if np.all(region.image):
        return "-"

    holes = count_holes(region)

    if holes == 2:  # "B" или "8"
        _, cx = region.centroid_local
        cx /= region.image.shape[1]
        return "B" if cx < 0.44 else "8"

    elif holes == 1:  # "A", "0", "P", "D"
        if count_vlines(region) > 1:
            # D и P
            return "P" if is_P(region) else "D"
        else:
            # A или 0
            cy, cx = region.centroid_local
            cx /= region.image.shape[1]
            cy /= region.image.shape[0]
            return "0" if abs(cx - cy) < 0.03 else "A"

    else:
        if count_vlines(region) >= 3:
            return "1"
        elif region.eccentricity < 0.45:
            return "*"
        else:
            inv = ~region.image
            dilated = binary_dilation(inv, np.ones((3, 3)))
            labeled = label(dilated, connectivity=1)
            n = np.max(labeled)
            if n == 2:
                return "/"
            elif n == 4:
                return "X"
            else:
                return "W"

    return "#"


path = Path(__file__).parent
image = imread(path / "symbols.png")[..., :3]
gray = rgb2gray(image)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

result = {}
out_dir = path / "out"
out_dir.mkdir(exist_ok=True)

for i, region in enumerate(regions):
    print(f"[{i + 1}/{len(regions)}]")
    symbol = recognize(region)
    result[symbol] = result.get(symbol, 0) + 1

    plt.imshow(region.image, cmap="gray")
    plt.title(symbol)

print("\nЧастотный словарь символов:")
for symbol, freq in sorted(result.items()):
    print(f"{symbol}: {freq}")
