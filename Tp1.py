import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
#############################################################################
def linha_vertical(ini_rows,ini_cols,fim_rows,fim_cols,rows,cols, corner, pontos, Contorno,edges,step ):
    u1 = 0
    v1 = 0
    for i in range(ini_rows,fim_rows,step):
         for j in range(ini_cols,fim_cols,step):
            if (corner[i, j] > 20 ):
                j_ant = j
                pontos.append([i,j])
                u = i
                v = j
                erro = 0
                u_velho = u

                while (erro < 3):
                    if (u <= rows and v <= cols):
                        if (edges[u, v] > 50 ):
                            Contorno[u, v] = 255
                            u = u + step
                            erro = 0

                            cv2.imshow('Contorno', Contorno)

                        elif (v < cols):

                            v = v + step
                            erro = erro + 1

                        else:
                            break
                pontos.append([u1, v1])

    return 0
#############################################################################

##############################################################################
def getpontos(frame, pontos):


    #frame = cv2.GaussianBlur(frame,(7,7),1,frame,1,cv2.BORDER_REFLECT)
    #frame = cv2.blur(frame, (8, 8))
    #resize
    scale_percent = 100  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
      # ponto no centro da figura


   # cv2.imshow("Rotacionado 45 graus", rotacionado)



    #gray = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    h1, w1 = frame.shape[:2]
    rows = math.floor(h1) - 1  # transforma em int
    cols = math.floor(w1) - 1  # transforma em int
    corner = np.zeros(shape=[h1,w1], dtype=np.uint8)


    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(gray, 100, 200, 3)

    gray[dst > 0.1 * dst.max()] = 255

    corner[dst > 0.1 * dst.max()] = 253



    #frame = cv2.resize(frame, dim, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    #corner = cv2.resize(corner, dim, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    #edges = cv2.resize(edges, dim, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    h1, w1 = frame.shape[:2]
    rows = math.floor(h1) - 1  # transforma em int
    cols = math.floor(w1) - 1  # transforma em int

    Contorno = np.zeros((h1, w1))



    ini_rows = 0
    ini_cols = 0
    fim_rows = rows-1
    fim_cols = cols-1
    step = 1
    linha_vertical(ini_rows,ini_cols,fim_rows,fim_cols,rows,cols, corner, pontos, Contorno,edges,step )



    cv2.imshow('Edges', edges)
    cv2.imshow('Contorno', Contorno)

    return 0

#############################################################################

###############################  M - A - I - N ##########################################
img = cv2.imread('alvo.jpg', 0)
imgplot = plt.imshow(img)

# capture frames from a camera
cap = cv2.VideoCapture('entrada.avi')
parada = 0
# loop runs if capturing has been initialized
while (1):

    # reads frames from a camera
    ret, frame = cap.read()

    #frame=cv2.copyMakeBorder(frame,80,80,0,0,0,frame,0)
    pontos = [0]
    #getpontos(frame, pontos)
    #pontos1 = pontos
    altura, largura = frame.shape[:2]
    rotacao = cv2.  getRotationMatrix2D((largura / 2, altura / 2), 90, 1.0)
    frame = cv2.warpAffine(frame, rotacao, (altura,largura))
    getpontos(frame,pontos)
    #pontos2 = pontos
    #print(rotacao)
    #print(pontos1)
   # print(pontos2)

# Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
       break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

