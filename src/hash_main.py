
from hash_comfusion_mat import ConfMat

def main():
    threshold = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                 ]

    mat = ConfMat(threshold)

    confMat, tpr, fpr = mat.resMat, mat.tpr, mat.fpr

    print(confMat)
    print(tpr)
    print(fpr)



if __name__ == "__main__":
    main()