import cv2
from difference_cal import Difference
import numpy as np
from confusion_mat import ConfMat

def main():
    threshold = [[0.1, 0.2, 0.3, 0.4, 0.45, 0.5],
                 [0.1, 0.2, 0.3, 0.35, 0.4, 0.45],
                 [0.1, 0.2, 0.3, 0.35, 0.4, 0.45],
                 [0.1, 0.15, 0.2, 0.23, 0.27, 0.3],
                 [0.01, 0.1, 0.15, 0.2, 0.25, 0.3],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                 ]

    mat = ConfMat(threshold)

    confMat, tpr, fpr = mat.resMat, mat.tpr, mat.fpr

    print(confMat)
    print(tpr)
    print(fpr)

if __name__ == "__main__":
    main()
