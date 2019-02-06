import cv2
from scipy import ndimage
import itertools
import time
from matplotlib.pyplot import cm 
from skimage import exposure
from vector import distance, pnt2line
import numpy as np

obj_id = -1
def naredni():
    global obj_id
    obj_id += 1
    return obj_id

def objektiPored(item, objekti):
    ret = []
    for o in objekti:
        udaljenost = distance(item['centar'], o['centar'])
        if(udaljenost<19):  # Parametar 20
            ret.append(o)
    return ret

def predict_konv(mreza, kontura):
    img_norm = kontura.reshape(1, 1, 28, 28).astype('float32')
    img_for_read = img_norm / 255
    predikcija = mreza.predict(img_for_read)
    predikcija = predikcija.reshape(1,10)
    num = int(np.argmax(predikcija))
    return num

def pronadji_cifre(snimak, xz1, yz1, xz2, yz2, xp1, yp1, xp2, yp2, mreza):
    presliZelenu = [] # Objekti koji su presli zelenu liniju
    presliPlavu = [] # Objekti koji su presli plavu liniju
    objekti = []

    cap = cv2.VideoCapture(snimak)

    while(cap.isOpened()):
    
        ret, img = cap.read()

        if not ret:
            break

        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img_bin = cv2.threshold(gray, 167, 255, 0) # 167 Parametar!
            kernel = np.ones((2,2),np.uint8)
            img_for_read = cv2.dilate(img_bin,kernel, iterations=2)
            labele, broj_labela = ndimage.label(img_for_read)
            brojevi = ndimage.find_objects(labele)
           
            for i in range(broj_labela):
                temp = brojevi[i]
                (centarX,centarY) = ((temp[1].stop + temp[1].start)/2,
                        (temp[0].stop + temp[0].start)/2)
                (duzinaX,duzinaY) = ((temp[1].stop - temp[1].start),
                        (temp[0].stop - temp[0].start))
                centarX = int(centarX)
                centarY = int(centarY)
                duzinaX = int(duzinaX)
                duzinaY = int(duzinaY)

                if(duzinaX>11 or duzinaY>11):
                    obj = {'centar':(centarX,centarY), 'centarX':centarX, 'centarY':centarY, 'duzinaX':duzinaX, 'duzinaY':duzinaY}
            
                    lst = objektiPored(obj, objekti)
                    u_okolini = len(lst)
                    if u_okolini == 0:
                        # Kreiranje novog objekta
                        obj['Z'] = False
                        obj['P'] = False
                        obj['id'] = naredni()
                        obj['centarX'] = centarX
                        obj['centarY'] = centarY
                        obj['duzinaX'] = duzinaX
                        obj['duzinaY'] = duzinaY
                        # Probati da nije for_read
                        isecena = img_for_read[int(centarY-duzinaY/2) : int(centarY+duzinaY/2), int(centarX-duzinaX/2) : int(centarX+duzinaX/2)]                        
                        uvecana = cv2.resize(isecena, (28, 28))
                        uvecana = cv2.erode(uvecana, np.ones((4,4),np.uint8))
                        obj['konture'] = []
                        obj['konture'].append(uvecana)
                        obj['broj_kontura'] = 1
                        objekti.append(obj)
                    elif u_okolini == 1:
                        lst[0]['centar'] = obj['centar']
                        lst[0]['centarX'] = obj['centarX']
                        lst[0]['centarY'] = obj['centarY']
                        lst[0]['duzinaX'] = obj['duzinaX']
                        lst[0]['duzinaY'] = obj['duzinaY']
                        isecena = img_for_read[int(centarY-duzinaY/2) : int(centarY+duzinaY/2), int(centarX-duzinaX/2) : int(centarX+duzinaX/2)]                        
                        uvecana = cv2.resize(isecena, (28, 28))
                        uvecana = cv2.erode(uvecana, np.ones((4,4),np.uint8))
                        if(lst[0]['broj_kontura'] < 950):
                            lst[0]['konture'].append(uvecana)
                        lst[0]['broj_kontura'] += 1

                    else: 
                        """isecena = img_for_read[int(centarY-duzinaY/2) : int(centarY+duzinaY/2), int(centarX-duzinaX/2) : int(centarX+duzinaX/2)]                        
                        uvecana = cv2.resize(isecena, (28, 28))
                        uvecana = cv2.erode(uvecana, np.ones((4,4),np.uint8))
                        pred = predict_konv(mreza, uvecana)
                        for x in lst:
                            isecena = img_for_read[int(x['centarY']-x['duzinaY']/2) : int(x['centarY']+x['duzinaY']/2), int(x['centarX']-x['duzinaX']/2) : int(x['centarX']+x['duzinaX']/2)]                        
                            uvecana = cv2.resize(isecena, (28, 28))
                            uvecana = cv2.erode(uvecana, np.ones((4,4),np.uint8))
                            x_pred = predict_konv(mreza, uvecana)
                            if pred == x_pred:
                                x['centar'] = obj['centar']
                                x['centarX'] = obj['centarX']
                                x['centarY'] = obj['centarY']
                                x['duzinaX'] = obj['duzinaX']
                                x['duzinaY'] = obj['duzinaY']
                                isecena = img_for_read[int(centarY-duzinaY/2) : int(centarY+duzinaY/2), int(centarX-duzinaX/2) : int(centarX+duzinaX/2)]                        
                                uvecana = cv2.resize(isecena, (28, 28))
                                uvecana = cv2.erode(uvecana, np.ones((4,4),np.uint8))
                                if(x['broj_kontura'] < 950):
                                    x['konture'].append(uvecana)
                                x['broj_kontura'] += 1
                                break
                        """

            for o in objekti:
                
                zelenaLinija = [(xz1,yz1), (xz2, yz2)]
                udaljenostZ, tackaZ, orientZ = pnt2line(o['centar'], zelenaLinija[0], zelenaLinija[1])
                if orientZ>0:
                    if(udaljenostZ<11): #11 parametar!
                         #cv2.line(img, tackaZ, o['centar'], (255, 0, 0), 1)
                        if o['Z'] == False:
                            o['Z'] = True
                            presliZelenu.append(o)

                plavaLinija = [(xp1,yp1), (xp2, yp2)]
                udaljenostP, tackaP, orientP = pnt2line(o['centar'], plavaLinija[0], plavaLinija[1])
                if orientP>0:
                    #cv2.line(img, pntP, o['centar'], (255, 0, 25), 1)
                    if(udaljenostP<15): # 15 parametar!
                        if o['P'] == False:
                            o['P'] = True
                            presliPlavu.append(o)
            
    cap.release()
    cv2.destroyAllWindows()
    return presliZelenu, presliPlavu



