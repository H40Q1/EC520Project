# EC520 Project
# Main

import sys
import cv2
from difference_cal import Difference
import numpy as np
from confusion_mat import ConfMat

def main():
    threshold = [[0.1, 0.2, 0.3, 0.4],
                 [0.2, 0.3, 0.4, 0.5],
                 [0.1, 0.2, 0.3, 0.4],
                 [0.1, 0.15, 0.2, 0.3],
                 [5, 10, 20, 40],
                 [0.001, 0.01, 0.1, 0.2],
                 [0.1, 0.15, 0.2, 0.3]
                 ]

    mat = ConfMat(threshold)
    tpr, fpr = mat.tpr, mat.fpr

if __name__ == "__main__":
    main()
