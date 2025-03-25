import numpy as np
import matplotlib.pyplot as plt

plus = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)

cross = np.array(
    [
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ]
)


def match(a, masks):
    for mask in masks:
        if a.shape == mask.shape and np.all((a != 0) == (mask != 0)):
            return True
    return False


def solve(mask1, mask2, image):
    e = 0
    masks = [mask1, mask2]
    mask_size = mask1.shape[0]
    offset = mask_size // 2

    for y in range(offset, image.shape[0] - offset):
        for x in range(offset, image.shape[1] - offset):
            sub = image[y - offset : y + offset + 1, x - offset : x + offset + 1]
            if match(sub, masks):
                e += 1
    return e


image = np.load("stars.npy")
print(f"Image shape: {image.shape}")

plt.imshow(image)
plt.show()

count = solve(plus, cross, image)
print(f"Number of matches found: {count}")
