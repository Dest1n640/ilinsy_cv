import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from pathlib import Path


def count_holes(region):
    image = np.pad(region.image, pad_width=1, mode="constant", constant_values=0)
    inverted = ~image
    labeled = label(inverted)
    return np.max(labeled) - 1


def count_vlines(region):
    return np.all(region.image, axis=0).sum()


def count_lgr_vlines(region):
    x = region.image.mean(axis=0) == 1
    return np.sum(x[: len(x) // 2]) > np.sum(x[len(x) // 2 :])


def recognize(region):
    if np.all(region.image):
        return "-"
    else:
        holes = count_holes(region)
        if holes == 2:  # 8 or B
            # vlines = count_vlines(region)
            _, cx = region.centroid_local
            cx /= region.image.shape[1]
            if cx < 0.44:
                return "B"
            return "8"
        elif holes == 1:  # A, 0
            cy, cx = region.centroid_local
            cx /= region.image.shape[1]
            cy /= region.image.shape[0]

            # Нахождение D и P
            if count_lgr_vlines(region):
                if cx > 0.4 or cy > 0.4:
                    return "D"
                else:
                    return "P"

            if abs(cx - cy) < 0.04:
                return "0"
            return "A"
        else:  # 1, *, /, X, W
            if count_vlines(region) >= 3:
                return "1"
            else:
                if region.eccentricity <= 0.4:
                    return "*"
                inv_image = ~region.image
                inv_image = binary_dilation(inv_image, np.ones((2, 2)))
                labeled = label(inv_image)
                match np.max(labeled):
                    case 2:
                        return "/"
                    case 4:
                        return "X"
                    case _:
                        return "W"
    return "#"


path = Path(__file__).parent
image = imread(path / "symbols.png")[..., :3]

gray = image.mean(axis=2)
binary = gray > 0
labeled = label(binary)
regions = regionprops(labeled)

result = {}
out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regions):
    print(f"{i + 1}/{len(regions)}")
    symbol = recognize(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(symbol)
    plt.imshow(region.image)
    plt.savefig(out_path / f"{i:03d}.png")


print(result)
