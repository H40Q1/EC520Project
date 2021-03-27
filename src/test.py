import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import math
from PIL import Image

def cosine(img1, img2):

    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    img1 = img1.resize((64, 64))
    img2 = img2.resize((64, 64))
    if grayscale:
        img1 = img1.convert('L')
        img2 = img2.convert('L')

    images = [img1, img2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = np.dot(a / a_norm, b / b_norm)
    d = 1 - res
    return round(d, 6)


def main():
    img1 = cv2.imread("../data/img1.jpg")
    img2 = cv2.imread("../data/img2.jpg")

    d = cosine(img1, img2)
    print(d)




    # cv2.imshow("edges",edges1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
