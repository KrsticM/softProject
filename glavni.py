import cv2
import sys
import numpy as np
from pronalazenje_cifara import pronadji_cifre
from konvoluciona_mreza import getKonvoluciona
from pronalazenje_prava import nadji_koordinate_zelene
from pronalazenje_prava import nadji_koordinate_plave

konvoluciona = getKonvoluciona()
suma = 0
  
f= open("out.txt","w+")
f.write("RA 107/2015 Milos Krstic\r")
f.write("file	sum\r")
f.close() 

for i in range(0, 10):
  snimak = 'snimci/video-' + str(i) + '.avi'
  cap = cv2.VideoCapture(snimak) # video je video path
  ret_val, frame = cap.read()

  linija_z = nadji_koordinate_zelene(frame) # mozda maksirati
  linija_p = nadji_koordinate_plave(frame) # mozda maksirati

  cap.release()
    
  brojeviZaOduzeti, brojeviZaSabrati = pronadji_cifre(snimak, linija_z[0], linija_z[1], linija_z[2], linija_z[3], linija_p[0], linija_p[1], linija_p[2], linija_p[3], konvoluciona)

  brojeviOd = []
  brojeviSa = []

  for b in brojeviZaOduzeti:
    postoji = False

    for br in brojeviZaSabrati:
      if b['id'] == br['id']:
        postoji = True

    if not postoji:
      brojeviOd.append(b)

  for b in brojeviZaSabrati:
    postoji = False

    for br in brojeviZaOduzeti:
      if b['id'] == br['id']:
        postoji = True

    if not postoji:
      brojeviSa.append(b)

  for b in brojeviOd:
    rez_pred = [0,0,0,0,0,0,0,0,0,0]
    for konture in b['konture']:
      img_norm = konture.reshape(1, 1, 28, 28).astype('float32')
      img_for_read = img_norm / 255
      predikcija = konvoluciona.predict(img_for_read)
      predikcija = predikcija.reshape(1,10)
      num = int(np.argmax(predikcija))
      rez_pred[num] += 1

    konacno = int(np.argmax(rez_pred))
    suma -= konacno
      

  for b in brojeviSa:
    rez_pred = [0,0,0,0,0,0,0,0,0,0]
    for konture in b['konture']:
      img_norm = konture.reshape(1, 1, 28, 28).astype('float32')
      img_for_read = img_norm / 255
      predikcija = konvoluciona.predict(img_for_read)
      predikcija = predikcija.reshape(1,10)
      num = int(np.argmax(predikcija))
      rez_pred[num] += 1
      
    konacno = int(np.argmax(rez_pred))
    suma += konacno
      

  f= open("out.txt","a+")
  f.write('video-' + str(i) + '.avi\t' + str(suma) + '\r')
  f.close()

  suma = 0