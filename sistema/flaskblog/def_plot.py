import os

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
from flaskblog.defs import *

def img_plot(x_new2,y_new,file_name,fig,ax,a_2,b_2):
    ruta = 'flaskblog/static/'+file_name
    imagen2 = cv2.imread(ruta, cv2.IMREAD_COLOR)
    img2 = rotate(imagen2, -90)

    dimensions = imagen2.shape
    height = imagen2.shape[0]
    width = imagen2.shape[1]
    ancho = int(width)
    altura = int(height)
    width = altura

    height = ancho
    dim = (width, height)
    resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    aa = ax.imshow(resized)

    altura = imagen2.shape[0]
    width = imagen2.shape[1]
    #y_grafo = y

    #y_inv = y[::-1]

    plt.plot(a_2, b_2, 'or')
    
    the_plot = plt.plot(x_new2, y_new)

    titulo= nombre_archivo(file_name)
    titulo_final=titulo+'pre.png'
    path = 'static/'
    plt.savefig(os.path.join(path, titulo_final))
    return titulo_final

def plot_rotate(imagen,ax,width,height):
    ruta = 'static/'+imagen
    imagen2 = cv2.imread(ruta, cv2.IMREAD_COLOR)
    img2 = rotate(imagen2, -270)

    dim = (width, height)
    resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
  

    titulo = nombre_archivo(imagen)
    titulo_final = titulo+'_gts.png'
    path = 'flaskblog/static/'
    
    status = cv2.imwrite(os.path.join(path, titulo_final), resized)

    return
'''
def plot_all(v,filename,ax):
    vector=v
    tamaño=len(vector)
    vector_pro=[]
    for i in range (0,tamaño):
        #print(vector[i])
        tam_i=len(vector[i])
        print(tam_i)
        #plt.plot(*sum(a, []), marker='o', color='r')
        #plt.show()
        for ii in range (0,tam_i):
            #print(vector[i][ii])
            alfa=vector[i][ii]
            vector_pro.append(alfa)
    print("ESTO RECIBOOO:",vector_pro)
    img = plt.imread("flaskblog/static/ex.png")
    #img2 = rotate(img, -270)


   
    lar=len(vector_pro)
    x_vector=[]
    y_vector=[]
    for i in range(0,lar):
        x_vector.append(vector_pro[i][0])
        y_vector.append(vector_pro[i][1])

    print("vector x",x_vector)
    print("vector y",y_vector)
    ax.imshow(img)
    plt.plot(x_vector,y_vector, marker='o')
    
    #818
    #1090
    plt.xlim(0, 818)
    plt.ylim(0,2000)
    titulo= nombre_archivo(filename)
    titulo_final=titulo+'_segundodebug.png'
    path = 'flaskblog/static/'
    plt.savefig(os.path.join(path, titulo_final))

    return titulo_final
'''