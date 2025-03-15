import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, binary_dilation

files = [
    "./files/wires1npy.txt",
    "./files/wires2npy.txt",
    "./files/wires3npy.txt",
    "./files/wires4npy.txt",
    "./files/wires5npy.txt",
    "./files/wires6npy.txt",
]

for file in files:
    data = np.load(file)
    print(f"{file}")
    labeled = label(data)  # маркируем изображение
    for i in range(1, np.max(labeled) + 1):
        result = binary_erosion(labeled == i, np.ones(3).reshape(3, 1))
        diff = np.max(label(result))
        if diff > 1:
            print(f"Wire {i} torne on {diff} parts")
        elif diff == 1:
            print(f"Wire {i} is full")
        else:
            print(f"Wire {i} was destroyed")

        plt.subplot(121)
        plt.imshow(data)
        plt.subplot(122)
        plt.imshow(result)
        plt.show()
