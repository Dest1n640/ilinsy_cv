import numpy as np
import matplotlib.pyplot as plt


external = np.diag([1, 1, 1, 1]).reshape(4, 2, 2)

internal = np.logical_not(external)

cross = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])


def match(a, masks):
    for mask in masks:
        if np.all((a != 0) == (mask != 0)):
            return True
    return False


def count_objects(image):
    E = 0
    for y in range(0, image.shape[0] - 1):
        for x in range(0, image.shape[1] - 1):
            sub = image[y : y + 2, x : x + 2]
            if match(sub, external):
                E += 1
            elif match(sub, internal):
                E -= 1
            elif match(sub, cross):
                E += 2
    return E / 4


image1 = np.load("example1.npy")
image2 = np.load("example2.npy")

plt.subplot(221)
plt.title("Image1")
plt.imshow(image1)
plt.subplot(222)
plt.title("Image2, first")
plt.imshow(image2[:, :, 0])
plt.subplot(223)
plt.title("image2, second")
plt.imshow(image2[:, :, 1])
plt.subplot(224)
plt.title("image2, third")
plt.imshow(image2[:, :, 2])


print(f"Image1 amount of figures: {count_objects(image1)}")
arr_img2 = [
    count_objects(image2[:, :, 0]),
    count_objects(image2[:, :, 1]),
    count_objects(image2[:, :, 2]),
]
result_image2 = sum(arr_img2)
print(f"Image2 amount of figures: {result_image2}")

plt.show()
