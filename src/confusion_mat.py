from difference_cal import Difference
from extract_images import extract
import cv2

class ConfMat:
    def __init__(self, threshold):
        self.n = len(threshold[0])  # n = number of thresholds
        self.numDiff = len(threshold)  # number of measures = 6
        self.threshold = threshold
        self.resMat = self.computeImg()  # 6*(4n) matrix
        self.tpr, self.fpr = self.computeROC()

    def computeImg(self):
        # 1) initialize resMat
        # in this case n = 6 , so:
        # resMat = [tp0, fp0, fn0, tn0 .. ] * 6 = matrix (6,24)
        #  each row comprises 6 confusion matrices (totally 24 elements) for 6 thresholds
        #  totally 6 rows, one row for each difference measures

        resMat = [[0]*(self.n*4) for _ in range(self.numDiff)]  # 6*(4n) matrix

        # 2) load data,
        # for each img in 700 candidates img, compare it to 100 original img
        # index i represents the index of candidate img
        # index j represents the index of original img each candidate img is comparing to

        for i in range(700):
            # code here: get candidate img i
            ori = ...

            # code here: determine if candidate is a copy, if it is, find its source img index
            if 0 <= i <= 499:
                src_index = i//5   # 5 distortions for each src img
            else:
                src_index = -1  # if not copy, assign src = -1

            for j in range(100):  # comparing to all original imgs
                # code here: get original img j
                candidate = ...

                # code here: resize candidate img i to the same size of original img
                dsize = (ori.shape(0), ori.shape(1))
                candidate = cv2.resize(candidate, dsize)

                # code here: compute differences by 6 measures
                diff = Difference(candidate, ori)
                diffList = diff.res

                # code here: use update method to update resMat
                self.update(j, src_index, diffList, resMat)

        # 3) return resMat
        return resMat

    def update(self, ori_index, src_index, diffList, resMat):
        # threshold shape(i,j)
        # resMat shape(i, j*4)
        # diffList shape (i, 1)
        for i in range(len(diffList)):
            for j in range(self.n):
                if diffList[i] <= self.threshold[i][j] and ori_index == src_index:
                    resMat[i][4 * j] += 1  # TP
                elif diffList[i] > self.threshold[i][j] and ori_index == src_index:
                    resMat[i][4 * j + 2] += 1  # FN
                elif diffList[i] <= self.threshold[i][j] and ori_index != src_index:
                    resMat[i][4 * j + 1] += 1  # FP
                else:
                    resMat[i][4 * j + 3] += 1  # TN

    def computeROC(self):
        tpr, fpr = [[0]*self.n for _ in range(self.numDiff)], [[0]*self.n for _ in range(self.numDiff)]
        # 6*6 matrix
        n = self.n
        for i in range(self.numDiff):
            for j in range(self.n):
                # index order : tp, fp, fn, tn
                tpr[i][j] = self.resMat[i][4*j] / (self.resMat[i][4*j] + self.resMat[i][4*j+2])  # tpr = tp/(tp+fn)
                fpr[i][j] = self.resMat[i][4*j+1] / (self.resMat[i][4*j+1] + self.resMat[i][4*j+3])  # fpr = fp/(fp+tn)

        return tpr, fpr








