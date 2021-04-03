# EC520 Project
# calculate 7 differences between two images

import cv2
import numpy as np
import SimpleITK as sitk
import math
from PIL import Image
from skimage.measure import compare_ssim


class Difference:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.height, self.width, self.channel = img1.shape

        self.img_diff = self.image_difference()  # Difference 1: Image difference
        self.rgb_hist = self.rgb_histogram()  # Difference 2: RGB histogram (16 bins)
        self.hsv_hist = self.hsv_histogram()  # Difference 3: HSV histogram (16+8+4 bins)
        self.hog_diff = self.hog_difference()  # Difference 4: Gradient Direction histogram (18 bins)
        self.hau_diff = self.hausdorff()  # Difference 5: Hausdorff Distance
        self.cos_diff = self.cosine()  # **Difference 6: Cosine similarity
        self.ssim_diff = self.ssim_measure()  # **Difference 7: SSIM
        # store in list
        self.res = [self.img_diff, self.rgb_hist, self.hsv_hist, self.hog_diff, self.hau_diff, self.cos_diff, self.ssim_diff]

#         self.inv_diff = self.invariant()  # Difference 7: Invariant Movement (?

    def image_difference(self):  # 1
        d = np.sum(abs(self.img1 - self.img2)) / (self.channel * self.width * self.height)
        return round(d / 255, 5)

    def rgb_histogram(self):  # 2
        hist1r = cv2.calcHist([self.img1], [0], None, [16], [0, 256])
        hist1g = cv2.calcHist([self.img1], [1], None, [16], [0, 256])
        hist1b = cv2.calcHist([self.img1], [2], None, [16], [0, 256])
        h1 = [hist1r, hist1g, hist1b]
        hist2r = cv2.calcHist([self.img2], [0], None, [16], [0, 256])
        hist2g = cv2.calcHist([self.img2], [1], None, [16], [0, 256])
        hist2b = cv2.calcHist([self.img2], [2], None, [16], [0, 256])
        h2 = [hist2r, hist2g, hist2b]
        tot = 0
        for i in range(self.channel):
            for j in range(16):
                tot += min(h1[i][j], h2[i][j])
        d = 1 - tot[0] / (self.channel * self.width * self.height)

        return round(d, 5)

    def hsv_histogram(self):  # 3
        img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2HSV)
        img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2HSV)
        hist1h = cv2.calcHist([img1], [0], None, [16], [0, 180])
        hist1s = cv2.calcHist([img1], [1], None, [8], [0, 256])
        hist1v = cv2.calcHist([img1], [2], None, [4], [0, 256])
        hist2h = cv2.calcHist([img2], [0], None, [16], [0, 180])
        hist2s = cv2.calcHist([img2], [1], None, [8], [0, 256])
        hist2v = cv2.calcHist([img2], [2], None, [4], [0, 256])
        h1 = [hist1h, hist1s, hist1v]
        h2 = [hist2h, hist2s, hist2v]
        tot = 0
        for i in range(self.channel):
            bin, _ = h1[i].shape
            for j in range(bin):
                tot += min(h1[i][j], h2[i][j])
        d = 1 - tot[0] / (self.channel * self.width * self.height)

        return round(d, 5)

    def hog_difference(self):  # 4
        img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        gx1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
        gy1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
        mag1, angle1 = cv2.cartToPolar(gx1, gy1, angleInDegrees=True)
        gx2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=1)
        gy2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=1)
        mag2, angle2 = cv2.cartToPolar(gx2, gy2, angleInDegrees=True)

        hist1 = [0] * 18
        hist2 = [0] * 18
        for i in range(self.height):
            for j in range(self.width):
                n1 = int(angle1[i, j]) // 20
                hist1[n1] += 1
                n2 = int(angle2[i, j]) // 20
                hist2[n2] += 1

        tot = 0
        for i in range(18):
            tot += min(hist1[i], hist2[i])
        d = 1 - tot / (self.width * self.height)
        return d

    def hausdorff(self):  # 5
        edges1 = cv2.Canny(self.img1, 25, 255, L2gradient=False)
        edges2 = cv2.Canny(self.img2, 25, 255, L2gradient=False)
        label1 = sitk.GetImageFromArray(edges1, isVector=False)
        label2 = sitk.GetImageFromArray(edges2, isVector=False)

        hausdorffcomputer = sitk.HausdorffDistanceImageFilter()  # does it need to be replaced by avgHausdorff?
        hausdorffcomputer.Execute(label1 > 0.5, label2 > 0.5)

        d = hausdorffcomputer.GetHausdorffDistance()
        d /= math.sqrt(self.width ^ 2 + self.height ^ 2)  # normalize: divided by the diagonal distance
        return round(d, 3)

    def cosine(self):  # 6
        img1 = Image.fromarray(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB))
        img1 = img1.resize((64, 64))
        img2 = img2.resize((64, 64))
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

    def ssim_measure(self):  # 7
        ssim = compare_ssim(self.img1, self.img2, multichannel=True)
        d = 1 - ssim
        return round(d, 3)

    """
    original methods 6 and 7...
    """

    def invariant(self):  # original 7 (may not use)
        x1, y1 = self.inv(self.img1)
        x2, y2 = self.inv(self.img2)
        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # this function is different from the one in paper

        return round(d)  # too large, need normalize?

    def inv(self, img):  # original 7 sub-method
        edges = cv2.Canny(img, 25, 255, L2gradient=False)
        totI, totJ, cnt = 0, 0, 0
        # find the centroid of the edge
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i, j] == 255:
                    totI += i
                    totJ += j
                    cnt += 1
        x_c, y_c = totI / cnt, totJ / cnt
        u20, u02, u11 = 0, 0, 0
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i, j] == 255:
                    u20 += (i - x_c) ** 2
                    u02 += (j - y_c) ** 2
                    u11 += (i - x_c) * (j - y_c)
        x = u20 + u02
        y = math.sqrt((u20 - u02) ** 2 + 4 * u11)
        return x, y
