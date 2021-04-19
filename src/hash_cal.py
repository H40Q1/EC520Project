# EC520 Project
# calculate 3 LSH differences between two images

import cv2
import numpy as np


class LSH:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.height, self.width, self.channel = img1.shape
        self.diff1 = self.campHash(self.ahash(img1), self.ahash(img2))
        self.diff2 = self.campHash(self.phash(img1), self.phash(img2))
        self.diff3 = self.campHash(self.dhash(img1), self.dhash(img2))
        self.res = [self.diff1, self.diff2, self.diff3]

    def ahash(self, image):
        image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        s = 0
        for i in range(8):
            for j in range(8):
                s = s + gray[i, j]
        avg = s / 64
        hash_str = ''
        for i in range(8):
            for j in range(8):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        result = ''
        for i in range(0, 64, 4):
            result += ''.join('%x' % int(hash_str[i: i + 4], 2))
        return result

    def phash(self, img):
        img = cv2.resize(img, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_roi = dct[0:8, 0:8]
        avreage = np.mean(dct_roi)
        phash_01 = (dct_roi > avreage) + 0
        phash_list = phash_01.reshape(1, -1)[0].tolist()
        hash = ''.join([str(x) for x in phash_list])
        return hash

    def dhash(self, image):
        image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dhash_str = ''
        for i in range(8):
            for j in range(8):
                if gray[i, j] > gray[i, j + 1]:
                    dhash_str = dhash_str + '1'
                else:
                    dhash_str = dhash_str + '0'
        result = ''
        for i in range(0, 64, 4):
            result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
        return result

    def campHash(self, hash1, hash2):
        n = 0
        if len(hash1) != len(hash2):
            return -1
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                n = n + 1
        return round(n/len(hash1), 3)

