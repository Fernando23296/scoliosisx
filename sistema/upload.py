import os

from flask import Flask, request, render_template, send_from_directory
from defs import *
from def_plot import *
from cnn import *
import cv2
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





@app.route("/")
def index():
    return render_template("upload.html")



@app.route("/upload", methods=["POST"])
def upload():

    target = os.path.join(APP_ROOT, 'static')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print("Save it to:", destination)
        file.save(destination)

        ruta = 'static/'+filename
        img = cv2.imread(ruta, cv2.IMREAD_COLOR)
        imagen2=img
        width = img.shape[1]
        print(width)
    print("*"*50)
    print("ESTO ES FILENAME:"+filename)
    print("*"*50)
    tipo=cnn(filename)
    print(tipo)
    if (tipo=='s'):
        division=12
    else:
        division=8
    print(division)
    height = img.shape[0]
    width = img.shape[1]
    height_original=int(height)
    width_original=int(width)
    qua = int(width/10)
    qua2 = int(qua*3)
    qua7 = int(qua*7)
    img[0:height, 0:qua2] = [0]
    img[0:height, qua7:width] = [0]

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 198, 300, cv2.THRESH_BINARY)[1]
    edges2 = feature.canny(thresh, sigma=3)
    skel = skeletonize(edges2)

    extracto = nombre_archivo(filename)


    nuevo_nombre = extracto+'_gts.png'
    
    path = 'static/'

    cv2.imwrite(os.path.join(path,nuevo_nombre), skel.astype(np.uint8)*255)
   

    extracto = nombre_archivo(filename)
    complemento = '_gts.png'
    titulo_final = extracto+complemento
    primer_debug=titulo_final
    print("*"*50)
    print("ESTO ES titulo final:"+titulo_final)
    print("*"*50)


    #______________________________________________________________________________________________


    img = cv2.imread(os.path.join(path,titulo_final))
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    ancho = int(width)
    altura = int(height)
    alfa = int(altura/division)
    cons = 0
    a = np.empty(((division+1), 50), dtype=object)

    for i in range(0, (division+1)):

        cons1 = cons
        cons2 = cons1+alfa
        image = img[cons1:cons2, 0:ancho]

        #convirtiendo a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #aplicando desenfoque gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #threshold?
        thresh = cv2.threshold(blurred, 60, 200, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        count = 1
        for c in cnts:
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                    cX, cY = 0, 0
            xx = str(cX)+","+str(cY)
            a[i][count] = [cX, cY]

            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
            #CIRCULO DE CENTRO
            cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
            #COORDENADAS
            cv2.putText(image, xx, (cX - 20, cY - 20),
                        #TIPO DE LETRA, COLOR?
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.imshow("Image", img)
            count = count+1

        cons = cons2
        i = i+1

    cero = [0, 0]
    b = [[cero if x is None else x for x in c] for c in a]
    #b = b[::-1]

    b = igualador(b,division,alfa)

    b = reemplazador(b, division)
    ax = np.zeros(shape=((division+1), 1), dtype=object)
    contador = 0

    lis_2 = []

    for i in range(0, (division+1)):
        a = limpio(b[i])
        lis_2.append(a)

    lis_3 = []
    lis_3 = rellenador(lis_2, division, ancho, altura)
    ax = seleccionador_kmeans(lis_3)

    #print("Lista: ", ax)

    ancho2 = int(ancho/2)
    axx = np.asarray(ax)
    bx = np.array([[ancho2, altura]])
    cx = np.concatenate((axx, bx), axis=0)
    dx = np.array([[ancho2, 0]])

    if (ax[0][1] < alfa):
        axx = np.concatenate((dx, cx), axis=0)
        axx = np.delete(axx, 1, 0)
        axx = np.array(axx.T)
    else:
        axx = np.array(cx.T)

    dim = (ancho, altura)
    resized = cv2.resize(imagen2, dim, interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots()
    
    l_x = axx[0]
    l_x = l_x.tolist()
    l_y = axx[1]
    l_y = l_y.tolist()

    #de aqui para abajo es todo girado
    y = axx[0]
    x = axx[1]

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


    #print("___"*20)
    pi1, pi2 = punto_inflexion(x, y)
    #print("+++"*20)
    a, b = max_min(x, y)
    a_2 = a[1:3]
    b_2 = b[1:3]
    #print(a)
    #print("___"*20)
    #print(b)


    z = np.polyfit(x, y, 5)
    f = np.poly1d(z)

    #print("Ecuacion bonita: ")
    #print(f)
    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    x_new2 = x_new[::-1]

    y_new = f(x_new2)
    y_new = y_new[::-1]

    pre=img_plot(x_new2,y_new,filename,fig,ax,a_2,b_2)
    plot_rotate(pre,ax,width_original,height_original)

    extracto2= nombre_archivo(filename)
    extracto21=extracto2+'pre'
    complemento = '_gts.png'
    titulo_final2 = extracto21+complemento
    print("esto es titulo final:"+titulo_final2)
    return render_template("complete.html", image_original=filename, image_name=titulo_final2, image_filter=primer_debug)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
