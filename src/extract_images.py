import cv2, os, random
import numpy as np


class extract:
    #This class can extract 11 images in total, including
    #Original * 1
    #Copies   * 5
    #Random images for each type * 5
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'\\data'#Current folder

    def get_all_image_names(self):
        original_index = self.root_path + r'\\original'
        #Read in all files and store them 
        #root = original_index, files = list for all filenames
        for root, dirs, files in os.walk(original_index):#Read in all files and store them 
            continue
        return files

    def extract_original(self, filename):
        original_path = self.root_path + r'\\original\\' + filename
        return cv2.imread(original_path)

    def extract_blurred(self, filename):
        blurred_path = self.root_path + r'\\blurred\\' + filename
        return cv2.imread(blurred_path)

    def extract_boarder(self, filename):
        boarder_path = self.root_path + r'\\boarder\\' + filename
        return cv2.imread(boarder_path)

    def extract_gauss_noise(self, filename):
        gauss_noise_path = self.root_path + r'\\gauss_noise\\' + filename
        return cv2.imread(gauss_noise_path)

    def extract_translation(self, filename):
        translation_path = self.root_path + r'\\translation\\' + filename
        return cv2.imread(translation_path)

    def extract_ycbcr(self, filename):
        ycbcr_path = self.root_path + r'\\ycbcr\\' + filename
        return cv2.imread(ycbcr_path)

    def extract_noncopy(self, filename):
        files = self.get_all_image_names()
        files.remove(filename)#Remove the selected image
        #read the original image's size
        original_index = self.root_path + r'\\original\\' + filename
        orig_img = cv2.imread(original_index)
        h, w = orig_img.shape[0:2]

        #Randomly select an image for each copy type
        blurred_name = files[random.randint(0, len(files)-2)]
        boarder_name = files[random.randint(0, len(files)-2)]
        gauss_noise_name = files[random.randint(0, len(files)-2)]
        translation_name = files[random.randint(0, len(files)-2)]
        ycbcr_name = files[random.randint(0, len(files)-2)]

        #Extract the randomly selected images and resize them
        blurred = cv2.resize(self.extract_blurred(blurred_name), (w, h), interpolation=cv2.INTER_LINEAR)
        boarder = cv2.resize(self.extract_boarder(boarder_name), (w, h), interpolation=cv2.INTER_LINEAR)
        gauss_noise = cv2.resize(self.extract_gauss_noise(gauss_noise_name), (w, h), interpolation=cv2.INTER_LINEAR)
        translation = cv2.resize(self.extract_translation(translation_name), (w, h), interpolation=cv2.INTER_LINEAR)
        ycbcr = cv2.resize(self.extract_ycbcr(ycbcr_name), (w, h), interpolation=cv2.INTER_LINEAR)

        return [blurred, boarder, gauss_noise, translation, ycbcr]
    
    def extract_copies(self, filename):
        blurred = self.extract_blurred(filename)
        boarder = self.extract_boarder(filename)
        gauss_noise = self.extract_gauss_noise(filename)
        translation = self.extract_translation(filename)
        ycbcr = self.extract_ycbcr(filename)
        return [blurred, boarder, gauss_noise, translation, ycbcr]
    
    def extract_all(self, filename):
        original = [self.extract_original(filename)]
        copy = self.extract_copies(filename)
        noncopy = self.extract_noncopy(filename)
        all = []
        all.extend(original)
        all.extend(copy)
        all.extend(noncopy)
        return all

def demo_extract_noncopy():
    ex = extract()
    noncopy = ex.extract_noncopy(filename='01.jpg')
    cv2.imshow('blurred', noncopy[0])
    cv2.imshow('boarder', noncopy[1])
    cv2.imshow('noise', noncopy[2])
    cv2.imshow('translation', noncopy[3])
    cv2.imshow('ycbcr', noncopy[4])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_extract_all_for_all():
    all = []
    ex = extract()
    for name in ex.get_all_image_names():
        all.append(ex.extract_all(name)) 
    return all