import socket
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


host = "84.237.21.36"
port = 5152
beat = None

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))

    while beat != b"yep":
        sock.send(b"get")
        bts = recvall(sock, 40002)

        beat = b"nope"

        im1 = np.frombuffer(bts[2:40002], dtype="uint8").reshape(bts[0], bts[1])

        binary = im1 > 0
        labeled = label(binary)
        regions = regionprops(labeled)
        if len(regions) == 2:
            pos1 = np.array(regions[0].centroid)
            pos2 = np.array(regions[1].centroid)
            result = np.sqrt(np.sum((pos1 - pos2) ** 2))

            sock.send(f"{result:.1f}".encode())
            print(sock.recv(10))
            #      plt.clf()
            #      plt.imshow(im1)
            #       plt.show()
            sock.send(b"beat")
            beat = sock.recv(10)
