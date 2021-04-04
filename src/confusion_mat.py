from difference_cal import Difference
from extract_images import extract

class ConfMat:
    def __init__(self, threshold):
        self.n = len(threshold[0])  # n = number of thresholds
        self.threshold = threshold
        self.resMat = self.computeImg()  # 7*(4n) matrix
        self.tpr, self.fpr = self.computeROC()

    def computeImg(self):
        # 1) initalize resMat
        # in this case n = 4, so:
        # resMat = [tp0, fp0, fn0, tn0, tp1, fp1, fn1, tn1, tp2, fp2, fn2, tn2, tp3, fp3, fn3, tn3] * 7
        #  each row comprises n confusion matrices for 7 thresholds
        #  totally 7 rows, one row for each difference
        # each confusion matrix sum up to 10 times of total number of images
        resMat = [[0]*(self.n*4) for _ in range(7)]  # 7*(4n) matrix

        # 2) load data,
        # iteratively get image list (size 11) from ..
        # at each iteration, use the self.computeDiff to update resMat
        extract_img = extract()
        for name in extract_img.get_all_image_names():
            img = extract_img.extract_all(name)
            self.computeDiff(img, resMat)
        # 3) return resMat
        return resMat

    def computeDiff(self, imgList, resMat):
        ori = imgList[0]
        for i in range(1, 11):
            diff = Difference(ori, imgList[i])
            diffList = diff.res  # list of diffs (size 7)
            for j in range(len(diffList)):
                for k in range(4):
                    if diffList[j] <= self.threshold[j][k] and i<6:
                        resMat[j][4 * k] += 1  # TP
                    elif diffList[j] > self.threshold[j][k] and i<6:
                        resMat[j][4 * k + 2] += 1  # FN
                    elif diffList[j] <= self.threshold[j][k] and i>=6:
                        resMat[j][4 * k + 1] += 1  # FP
                    else:
                        resMat[j][4 * k + 3] += 1  # TN

    def computeROC(self):
        tpr, fpr = [[0]*self.n for _ in range(7)], [[0]*self.n for _ in range(7)]
        # 7*n matrix
        n = self.n
        for i in range(len(self.resMat)):
            for j in range(self.n):
                # index order : tp, fp, fn, tn
                tpr[i][j] = self.resMat[i][4*j] / (self.resMat[i][4*j] + self.resMat[i][4*j+2])  # tpr = tp/(tp+fn)
                fpr[i][j] = self.resMat[i][4*j+1] / (self.resMat[i][4*j+1] + self.resMat[i][4*j+3])  # fpr = fp/(fp+tn)

        return tpr, fpr