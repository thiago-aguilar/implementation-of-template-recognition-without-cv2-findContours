import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

import numpy as np

def rotateImage(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

img = cv2.imread('alvo.jpg', 0)
alvo1=np.copy(img)
alvo2= rotateImage(alvo1,90)
alvo3= rotateImage(alvo2,90)
alvo4= rotateImage(alvo3,90)
cv2.imshow('image', img)
cv2.imshow('alvo1', alvo1)
cv2.imshow('alvo2', alvo2)
cv2.imshow('alvo3', alvo3)
cv2.imshow('alvo4', alvo4)
imgplot = plt.imshow(img)

# capture frames from a camera
cap = cv2.VideoCapture('entrada.avi')

tamXimg = img.shape[1]
tamYimg = img.shape[0]
witdh = cap.get(3)  # float
height = cap.get(4)  # float
rows = math.floor(height)  # transforma em int
cols = math.floor(witdh)  # transforma em int
rows = rows
cols = cols

# loop runs if capturing has been initialized
while (1):

    # reads frames from a camera
    ret, frame = cap.read()
    # scale_percent = 100  # percent of original size
    # width = int(frames.shape[1] * scale_percent / 100)
    # height = int(frames.shape[0] * scale_percent / 100)
    # rows = width
    # cols = height
    # dim = (width, height)
    # # resize image
    # frame = cv2.resize(frames, dim, interpolation=cv2.INTER_AREA)
    # converting BGR to HSV
  #  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # define range of red color in HSV
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    # create a red HSV colour boundary and
    # threshold HSV image
#    mask = cv2.inRange(hsv, lower_red, upper_red)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Bitwise-AND mask and original image
 #   res = cv2.bitwise_and(frame, frame, mask=mask)

    # Display an original image
    cv2.imshow('Original', frame)
    original=np.copy(frame)
    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(frame, 150, 200)
    frame[dst > 0.005 * dst.max()] = [0, 0, 255]
    quinas=[]
    pontos=[]
    contaQuinas=0
    contador=0
    homografias=[]
    for i in range(cols-1):
        for j in range(rows-1):
            if (np.array_equal(255, edges[j, i])) and np.array_equal([0,0,255], frame[j, i]): #verifica se chegou em pixel que é borda e quina
                x=i
                y=j
                cv2.imshow('Corners', frame)
                lastx=0
                lasty=0
                lastlastx=-1
                lastlasty=-1
                sair=1
                contaQuinas=1
                pontos.append([y, x])  # se acabou de entrar, guarda o ponto inicial
                quinas.append([y, x])
                contador=0
                while sair==1:
                    contador=contador+1
                    if lastx<=x and lasty >=y :
                        if (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y))and((lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty= y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y - 1)))and((lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x])) and ((lastx != x) or (lasty != (y - 1)))and((lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x+1])) and ((lastx != (x + 1)) or (lasty != (y - 1)))and((lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x+1 ])) and ((lastx != (x + 1)) or (lasty != y))and((lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx=lastx
                            lastlasty=lasty
                            lasty=y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and ((lastx != (x + 1)) or (lasty != (y + 1))) and ((lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y+1, x])) and ((lastx != x) or (lasty != (y + 1)))and ((lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx=x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y + 1)))and((lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1

                    elif lastx<=x and lasty <y :
                        if (np.array_equal(255, edges[y-1, x])) and ((lastx != x) or (lasty != (y - 1)))and((lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x+1])) and ((lastx != (x + 1)) or (lasty != (y - 1)))and((lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x+1 ])) and ((lastx != (x + 1)) or (lasty != y))and((lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx=lastx
                            lastlasty=lasty
                            lasty=y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and ((lastx != (x + 1)) or (lasty != (y + 1))) and ((lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y+1, x])) and ((lastx != x) or (lasty != (y + 1)))and ((lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx=x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y + 1)))and((lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y))and((lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty= y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y - 1)))and((lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1

                    elif lastx>x and lasty>=y :
                        if (np.array_equal(255, edges[y+1, x])) and ((lastx != x) or (lasty != (y + 1)))and ((lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx=x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y + 1)))and((lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y))and((lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty= y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y - 1)))and((lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x])) and ((lastx != x) or (lasty != (y - 1)))and((lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x+1])) and ((lastx != (x + 1)) or (lasty != (y - 1)))and((lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x+1 ])) and ((lastx != (x + 1)) or (lasty != y))and((lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx=lastx
                            lastlasty=lasty
                            lasty=y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and ((lastx != (x + 1)) or (lasty != (y + 1))) and ((lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1

                    elif lastx>=x and lasty<y :
                        if (np.array_equal(255, edges[y, x+1 ])) and ((lastx != (x + 1)) or (lasty != y))and((lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx=lastx
                            lastlasty=lasty
                            lasty=y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and ((lastx != (x + 1)) or (lasty != (y + 1))) and ((lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y+1, x])) and ((lastx != x) or (lasty != (y + 1)))and ((lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx=x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y + 1)))and((lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y))and((lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty= y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and ((lastx != (x - 1)) or (lasty != (y - 1)))and((lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x])) and ((lastx != x) or (lasty != (y - 1)))and((lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y-1, x+1])) and ((lastx != (x + 1)) or (lasty != (y - 1)))and((lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                    if (x!= i or y != j) and  np.array_equal([0,0,255], frame[y, x]) and contador>20:
                        contaQuinas=contaQuinas+1
                        cv2.imshow('Corners', frame)
                        quinas.append([y,x])
                        contador=0
                    if x==i and y==j:
                        sair=0
                        # if contaQuinas == 4:
                        #     homografias.append(quinas)
                        # break
                    if x < 0 or y < 0 or x>=(cols-1) or y>=(rows-1): sair=0
                    if [y, x] in pontos:
                        #contaQuinas=0
                        sair=0

                    pontos.append([y,x]) #
                if contaQuinas>=4 and [y,x] in pontos :
                    homografias.append(quinas)

                quinas=[]
                sair=1

    #HOMOGRAFIA------------------------------------------
    for i in homografias:
       # value1=i
        #aux1=np.array([[i[0][1],i[0][0]],[i[1][1],i[1][0]],[i[2][1],i[2][0]],[i[3][1],i[3][0]]])
        aux1 = np.array([[i[0][1], i[0][0]],[i[3][1], i[3][0]], [i[2][1], i[2][0]], [i[1][1], i[1][0]] ])
        aux2=np.array([[0,0],[0,tamXimg],[tamYimg,tamXimg],[tamYimg,0]])
        h, status = cv2.findHomography(aux1, aux2, cv2.RANSAC,5)
        im_dst = cv2.warpPerspective(original, h, (tamYimg, tamXimg))
        cv2.imshow('HOMOGRAFIA', im_dst)
#
        # value1 = homografias[0]
        # teste1 = np.array([value1])
        # teste2 = np.array([[0,0],[0,tamYimg],[tamXimg,tamYimg],[tamXimg,0]])
        # h, status = cv2.findHomography(teste1,teste2)
        # im_dst = cv2.warpPerspective( frame , h, (tamXimg,tamYimg))


    # Display edges in a frame
    cv2.imshow('Edges', edges)
    cv2.imshow('Corners', frame)
    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

### Close the window
cap.release()
#
# De-allocate any associated memory usage
cv2.destroyAllWindows()