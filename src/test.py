from difference_cal import Difference
import cv2, os
from skimage.metrics import structural_similarity#for skimage=0.18.x

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'\\data'
path1 = root_path + r'\\original\\01.jpg'
path2 = root_path + r'\\blurred\\01.jpg'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
diff = Difference(img1, img2)
#diffList = diff.res