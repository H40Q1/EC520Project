# EC520 Project
# Main

import sys
import cv2
from difference_cal import Difference
import numpy as np


def main():
    # data
    path1 ="../data/img1.jpg"  # test case path
    path2 ="../data/img2.jpg"

    img1 = cv2.imread(path1)  # test case (432, 768, 3)
    img2 = cv2.imread(path2)

    diff = Difference(img1, img2)

    # print all 7 differences
    print("Difference-1 Image Difference :", diff.img_diff)
    print("Difference-2 RGB Histogram Difference :", diff.rgb_hist)
    print("Difference-3 HSV Histogram Difference :", diff.hsv_hist)
    print("Difference-4 Gradient Direction Histogram Difference :", diff.hog_diff)
    print("Difference 5: Hausdorff Distance :", diff.hau_diff)
    print("Difference 6: Cosine Similarity :", diff.cos_diff)
    print("Difference 7: Structural Similarity Index Measure :", diff.ssim_diff)

    # print("Difference 6: Local Edge Distance :", diff.loc_diff)
    # print("Difference 7: Invariant Movement Distance :", diff.inv_diff)



if __name__ == "__main__":
    main()
