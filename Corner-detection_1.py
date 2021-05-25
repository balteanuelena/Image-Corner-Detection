import pandas as pd
import cv2
import numpy as np

def Harris_Corner_Detection(image):
    img = cv2.imread(image)

    #Ne arata imaginea originala
    cv2.imshow('Imaginea originala', img)

    #Facem imaginea sa fie doar in nuante de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gri = np.float32(gri)

    #Metoda lui Harris de detectare a colturilor
    dst = cv2.cornerHarris(gri, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    colturi = cv2.cornerSubPix(gri, np.float32(centroids), (5, 5), (-1, -1), criteria)

    for i in range(1, len(colturi)):
        print(colturi[i])

    img[dst > 0.1 * dst.max()] = [0, 0, 255]

    for i in colturi:
        i[0] = np.round(i[0])
        i[1] = np.round(i[1])

    # Creaza cadrul de date Pandas pe care il returnam
    df = pd.DataFrame(data=colturi, columns=["x", "y"])

    # Ne arata imaginea modificata
    cv2.imshow("Imaginea modificata", dst)

    # Debug
    cv2.waitKey()

    print(df)
    return df


if __name__ == "__main__":
    Harris_Corner_Detection("FormeGeometrice.png")

# Returneaza pur si simplu un cadru de date 'Pandas'
# Codul foloseste metoda lui Harris de detectare a colturilor
# Pentru realizarea codului am folosit librarile: Pandas (care vine la pachet cu Numpy) si opencv-python
# Cu aceasta metoda, la rularea codului, putem vedea in terminal coordonatele X si Y pentru colturile detectate
