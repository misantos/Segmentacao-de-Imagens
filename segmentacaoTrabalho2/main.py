import sys
import numpy as np
import cv2

def calculaMediaX(fImage):
    mediaX =0
    for i in range(len(fImage)):
        for j in range(len(fImage[0])):
            mediaX += (j+1)*fImage[i,j]
    mediaX = mediaX/m00
    return mediaX

def calculaMediaY(fImage):
    mediaY =0
    for i in range(len(fImage)):
        for j in range(len(fImage[0])):
            mediaY += (i+1)*fImage[i,j]
    mediaY = mediaY/m00
    return mediaY

def mediaInvariancia(p, q, fImage):
    count = 0
    for i in range(len(fImage)):
        for j in range(len(fImage[0])):
            count += ((i+1)-m10m00)**p * ((j+1)-m01m00)**q * fImage[i,j]
    return count

def moment(p, q, fImage):
    lamda = ((p+q)/2) + 1
    media = mediaInvariancia (p, q, fImage)
    return media / ((m00) **lamda)


file_name = input('Insira o nome da imagem que deseja abrir: ')
file = open("SegmentacaoMomentosInvariantes.csv", "a")

imagem = cv2.imread(file_name)
original = imagem.copy()
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
canny = cv2.Canny(blurred, 120, 255, 1)
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=7)
opening = cv2.morphologyEx(dilate,cv2.MORPH_OPEN,kernel, iterations = 2)

#imagem para fazer o crop do contorno
dilateCrop = cv2.dilate(canny, kernel, iterations=1)
openingCrop = cv2.morphologyEx(dilateCrop,cv2.MORPH_OPEN,kernel, iterations = 1)
ret,threshCrop = cv2.threshold(openingCrop,127,255,cv2.THRESH_BINARY_INV)

cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

image_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if((w * h) > 102000):
        perimeter = cv2.arcLength(c,True)
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,0,255), 2)
        crop = original[y:y+h, x:x+w]
        imageFloat = crop.astype(float)
        if len(np.shape(imageFloat)) == 3:
            imageFloat = imageFloat[:,:,0]
        imageFloat = imageFloat / 255

        m00 = np.sum(imageFloat)
        m10m00 = calculaMediaX(imageFloat)
        m01m00 = calculaMediaY(imageFloat)

        eta1_1 = moment(1,1,imageFloat)
        eta2_0 = moment(2,0,imageFloat)
        eta0_2 = moment(0,2,imageFloat)

        moment1 = eta2_0 + eta0_2
        moment2 = ((eta2_0 + eta0_2)**2) + (4*(eta1_1**2))
        file.write(file_name)
        file.write(',')
        file.write('Numero da folha: ')
        file.write(str(image_number))
        file.write(',')
        file.write('Perimetro: ')
        file.write(str(perimeter))
        file.write(',')
        file.write('Momento Invariante 1: ')
        file.write(str(moment1))
        file.write(',')
        file.write('Momento Invariante 2: ')
        file.write(str(moment2))
        file.write('\n')

        cropContorno = threshCrop[y:y+h, x:x+w]
        cv2.imwrite(file_name[:-4] + '-' + str(image_number) + '.png', crop)
        cv2.imwrite((file_name[:-4] + '-' + str(image_number) + '-P.png'), cropContorno)
        image_number += 1

print('Numero das folhas: ',image_number)
