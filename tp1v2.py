import cv2
from matplotlib import pyplot as plt
import math
import numpy as np

videoFinal = []
def retorna_final(homografias,original,final):
# HOMOGRAFIA------------------------------------------
    for i in homografias:
        # value1=i
        # aux1=np.array([[i[0][1],i[0][0]],[i[1][1],i[1][0]],[i[2][1],i[2][0]],[i[3][1],i[3][0]]])
        aux1 = np.array([[i[0][1], i[0][0]], [i[3][1], i[3][0]], [i[2][1], i[2][0]], [i[1][1], i[1][0]]])
        aux2 = np.array([[0, 0], [0, tamXimg], [tamYimg, tamXimg], [tamYimg, 0]])
        h, status = cv2.findHomography(aux1, aux2)
        im_dst = cv2.warpPerspective(original, h, (tamYimg, tamXimg))
        cv2.imshow('HOMOGRAFIA', im_dst)

        if METODO == 0:
            erro1 = compara1(im_dst, alvo1)
            erro2 = compara1(im_dst, alvo2)
            erro3 = compara1(im_dst, alvo3)
            erro4 = compara1(im_dst, alvo4)

            if erro1 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 255, 0), 2)
            if erro2 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 0, 255), 2)
            if erro3 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 255, 0), 2)
            if erro4 <= 12000:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 0, 0), 2)
        else:
            erro1 = compara2(im_dst, alvo1)
            erro2 = compara2(im_dst, alvo2)
            erro3 = compara2(im_dst, alvo3)
            erro4 = compara2(im_dst, alvo4)

            if erro1 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 255, 0), 2)
            if erro2 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 0, 255), 2)
            if erro3 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 0, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (0, 255, 0), 2)
            if erro4 >= 0.75:
                final = cv2.line(final, (i[0][1], i[0][0]), (i[3][1], i[3][0]), (0, 255, 0), 2)
                final = cv2.line(final, (i[3][1], i[3][0]), (i[2][1], i[2][0]), (0, 0, 255), 2)
                final = cv2.line(final, (i[2][1], i[2][0]), (i[1][1], i[1][0]), (255, 255, 0), 2)
                final = cv2.line(final, (i[1][1], i[1][0]), (i[0][1], i[0][0]), (255, 0, 0), 2)
    return final

def encontrar_homografia(homografias,edges,frame,pontos,quinas,cols,rows):
    for i in range(cols - 1):
        for j in range(rows - 1):
            if (np.array_equal(255, edges[j, i])) and np.array_equal([0, 0, 255], frame[j, i]):  # verifica se chegou em pixel que é borda e quina
                x = i
                y = j

                lastx = 0
                lasty = 0
                lastlastx = -1
                lastlasty = -1
                sair = 1
                contaQuinas = 1
                pontos.append([y, x])  # se acabou de entrar, guarda o ponto inicial
                quinas.append([y, x])
                contador = 0
                while sair == 1:
                    contador = contador + 1
                    if lastx <= x and lasty >= y:
                        if (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y)) and (
                                (lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x])) and ((lastx != x) or (lasty != (y - 1))) and (
                                (lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x + 1])) and ((lastx != (x + 1)) or (lasty != y)) and (
                                (lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lasty = y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x])) and ((lastx != x) or (lasty != (y + 1))) and (
                                (lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1

                    elif lastx <= x and lasty < y:
                        if (np.array_equal(255, edges[y - 1, x])) and ((lastx != x) or (lasty != (y - 1))) and (
                                (lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x + 1])) and ((lastx != (x + 1)) or (lasty != y)) and (
                                (lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lasty = y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x])) and ((lastx != x) or (lasty != (y + 1))) and (
                                (lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y)) and (
                                (lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1

                    elif lastx > x and lasty >= y:
                        if (np.array_equal(255, edges[y + 1, x])) and ((lastx != x) or (lasty != (y + 1))) and (
                                (lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y)) and (
                                (lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x])) and ((lastx != x) or (lasty != (y - 1))) and (
                                (lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y, x + 1])) and ((lastx != (x + 1)) or (lasty != y)) and (
                                (lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lasty = y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1

                    elif lastx >= x and lasty < y:
                        if (np.array_equal(255, edges[y, x + 1])) and ((lastx != (x + 1)) or (lasty != y)) and (
                                (lastlastx != (x + 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lasty = y
                            lastx = x
                            x = x + 1
                        elif (np.array_equal(255, edges[y + 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x])) and ((lastx != x) or (lasty != (y + 1))) and (
                                (lastlastx != x) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            y = y + 1
                        elif (np.array_equal(255, edges[y + 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y + 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y + 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y + 1
                        elif (np.array_equal(255, edges[y, x - 1])) and ((lastx != (x - 1)) or (lasty != y)) and (
                                (lastlastx != (x - 1)) or (lastlasty != y)):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                        elif (np.array_equal(255, edges[y - 1, x - 1])) and (
                                (lastx != (x - 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x - 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x - 1
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x])) and ((lastx != x) or (lasty != (y - 1))) and (
                                (lastlastx != x) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x
                            y = y - 1
                        elif (np.array_equal(255, edges[y - 1, x + 1])) and (
                                (lastx != (x + 1)) or (lasty != (y - 1))) and (
                                (lastlastx != (x + 1)) or (lastlasty != (y - 1))):
                            lastlastx = lastx
                            lastlasty = lasty
                            lastx = x
                            lasty = y
                            x = x + 1
                            y = y - 1
                    if (x != i or y != j) and np.array_equal([0, 0, 255], frame[y, x]) and contador > 20:
                        contaQuinas = contaQuinas + 1
                        # cv2.imshow('Corners', frame)
                        quinas.append([y, x])
                        contador = 0
                    if x == i and y == j:
                        sair = 0
                        # if contaQuinas == 4:
                        #     homografias.append(quinas)
                        # break
                    if x < 0 or y < 0 or x >= (cols - 1) or y >= (rows - 1): sair = 0
                    if [y, x] in pontos:
                        # contaQuinas=0
                        sair = 0

                    pontos.append([y, x])  #
                if contaQuinas >= 4 and [y, x] in pontos:
                    homografias.append(quinas)

                quinas = []
                sair = 1
    return homografias

def compara1(imageA, imageB):
    # implementando metodo meanSquaredError
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def compara2(imageA, imageB):
    # implementando metodo CCORR

    numerador = np.sum((imageA.astype("float") * imageB.astype("float")))
    denominador1 = np.sum(imageA.astype("float") ** 2)
    denominador2 = np.sum(imageB.astype("float") ** 2)
    denominador = math.sqrt(denominador1 * denominador2)
    err = numerador / denominador
    # err = numerador
    return err


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
METODO = 1  # 0 PARA MeanSquaredError | 1 para CCORR
imgplot = plt.imshow(img)
crop_img = img[:, 3:-4]

alvo1 = np.copy(crop_img)
alvo2 = rotateImage(alvo1, 90)
alvo3 = rotateImage(alvo2, 90)
alvo4 = rotateImage(alvo3, 90)

# capture frames from a camera
cap = cv2.VideoCapture('entrada.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

tamXimg = alvo1.shape[1]
tamYimg = alvo1.shape[0]
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
    if not ret: break

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
    # cv2.imshow('Original', frame)
    original = np.copy(frame)
    final = np.copy(original)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(frame, 150, 250)
    frame[dst > 0.0005 * dst.max()] = [0, 0, 255]
    quinas = []
    pontos = []
    contaQuinas = 0
    contador = 0
    homografias = []
    homografias = encontrar_homografia(homografias, edges, frame, pontos, quinas, cols, rows)
    final = retorna_final(homografias, original, final)
    cv2.imshow('FINAL', final)
    out.write(final)
    # Display edges in a frame
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Corners', frame)
    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

### Close the window
cap.release()
out.release()

#
# De-allocate any associated memory usage
cv2.destroyAllWindows()
