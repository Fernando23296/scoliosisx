from scipy.ndimage import rotate
from scipy.misc import imread, imshow
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import cv2
import imutils
from operator import is_not
from functools import partial
from pylab import *
from random import *
from sklearn.cluster import KMeans
from matplotlib import transforms
import scipy.ndimage.morphology as morp
from skimage import feature


def nombre_archivo(na):
        punto = na.find(".")
        b = ''
        for i in range(0, punto):
                b += str(na[i])
        return b


def skeletonize(img):

    struct = np.array([[[[0, 0, 0], [0, 1, 0], [1, 1, 1]],
                        [[1, 1, 1], [0, 0, 0], [0, 0, 0]]],

                       [[[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        [[0, 1, 1], [0, 0, 1], [0, 0, 0]]],

                       [[[0, 0, 1], [0, 1, 1], [0, 0, 1]],
                           [[1, 0, 0], [1, 0, 0], [1, 0, 0]]],

                       [[[0, 0, 0], [0, 1, 1], [0, 1, 0]],
                           [[1, 1, 0], [1, 0, 0], [0, 0, 0]]],

                       [[[1, 1, 1], [0, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [1, 1, 1]]],

                       [[[0, 1, 0], [0, 1, 1], [0, 0, 0]],
                           [[0, 0, 0], [1, 0, 0], [1, 1, 0]]],

                       [[[1, 0, 0], [1, 1, 0], [1, 0, 0]],
                           [[0, 0, 1], [0, 0, 1], [0, 0, 1]]],

                       [[[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 1], [0, 1, 1]]]])

    img = img.copy()
    last = ()
    while np.any(img != last):
        last = img
        for s in struct:
            img = np.logical_and(img, np.logical_not(
                morp.binary_hit_or_miss(img, *s)))
    return img


def igualador(l,division,alfa):
    contador = 1
    for i in range(1, (division+1)):
        for ii in range(0, 50):
            l[i][ii][1] += alfa*contador
        contador = contador+1
    return l


def reemplazador(l, div):
    #AQUI SE CAMBIO EL 1 DEL RANGE, ANTES ERA 2-------------------------------------
    for u in range(0, (div+1)):
        for uu in range(0, 50):
            if (l[u][uu][0] == 0):
                l[u][uu] = None
            else:
                pass
    return l


def limpio(l):
    tam = len(l)
    a = []
    for i in range(0, tam):
        if(l[i] != None):
            a.append(l[i])
    return a

def rellenador(l, div, ancho, largo):
    tam = len(l)
    largo = int(largo/div)
    ancho = int(ancho/2)
    for i in range(0, tam):
        if (l[i] == []):
            relleno = [ancho, largo*i]
            l[i].append(relleno)
        else:
            pass
    return l



def seleccionador(l):
    a = []
    tam = len(l)
    for i in range(0, tam):
        tam1 = (len(l[i])-1)

        a.append(l[i][randint(0, tam1)])
    return a


def seleccionador_kmeans(l):
    tam = len(l)
    xpro = []
    for i in range(0, tam):

        tam1 = len(l[i])
        x1 = np.array([])
        x1 = l[i]
        tam2 = len(x1)
        if (tam2 == 1):
            k = 1

            kmeans = KMeans(k)
            # Fitting the input data
            kmeans = kmeans.fit(x1)
            # Getting the cluster labels
            labels = kmeans.predict(x1)
            # Centroid values
            centroids = kmeans.cluster_centers_
            # Comparing with scikit-learn centroids
            #print(C)  # From Scratch
            xpro.append(centroids[0])
        else:
            k = 2

            kmeans = KMeans(k)
            # Fitting the input data
            kmeans = kmeans.fit(x1)
            # Getting the cluster labels
            labels = kmeans.predict(x1)
            # Centroid values
            centroids = kmeans.cluster_centers_
            # Comparing with scikit-learn centroids
            #print(C)  # From Scratch
            xpro.append(centroids[randint(0, 1)])
    return xpro


def max_min(x, y):
    z = np.polyfit(x, y, 5)
    #print(z)
    #print('.-.-'*30)
    f = np.poly1d(z)
    #print("Ecuacion bonita: ")
    #print(f)
    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)

    x_new2 = x_new[::-1]

    y_new = f(x_new2)
    y_new = y_new[::-1]  # LO GIRAMOS
    #a=f*10000000000

    prim = f.deriv(1)
    segu = f.deriv(2)
    lroot = np.roots(prim)
    lroot2 = []
    for i in lroot:
        lroot2.append(segu(i))
    lroot3 = []
    for ii in lroot:
        lroot3.append(f(ii))

    xdev = lroot[::-1]
    ydev = lroot3

    return xdev, ydev


def punto_inflexion(x, y):
    z = np.polyfit(x, y, 5)
    f = np.poly1d(z)
    print('ecuacion bonita')
    print(f)
    segu = f.deriv(2)
    print('segunda derivada', segu)
    lrootx = np.roots(segu)
    print("esto es lrootx: ", lrootx)
    lrooty = []
    for i in lrootx:
        lrooty.append(f(i))
    print("esto es lrooty: ", lrooty)
    return lrootx, lrooty
