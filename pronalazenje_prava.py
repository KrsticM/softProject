import cv2
import numpy as np


def nadji_koordinate_plave(frame):
   rgb_slika = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   img_ero = cv2.erode(rgb_slika, kernel, iterations=1)
   img_open = cv2.dilate(img_ero, kernel, iterations=1)

   plavi_kanal = img_open.copy()
   plavi_kanal[:, :, 0] = 0
   plavi_kanal[:, :, 1] = 0

   frame_gray = cv2.cvtColor(plavi_kanal, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(frame_gray,0,50)
   koordinate_plave = cv2.HoughLinesP(edges, rho = 1, theta = np.pi / 180, threshold = 110, minLineLength = 100, maxLineGap = 35)

   for plavePrave in koordinate_plave:
      for x1,y1,x2,y2 in plavePrave:
        return int(x1), int(y1), int(x2), int(y2)
 
def nadji_koordinate_zelene(frame):
   rgb_slika = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

   img_ero = cv2.erode(rgb_slika, kernel, iterations=1)
   img_open = cv2.dilate(img_ero, kernel, iterations=1)

   zeleni_kanal = img_open.copy()
   zeleni_kanal[:, :, 0] = 0
   zeleni_kanal[:, :, 2] = 0

   frame_gray = cv2.cvtColor(zeleni_kanal, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(frame_gray,0,50)

   koordinate_zelene = cv2.HoughLinesP(edges, rho = 1, theta = np.pi / 180, threshold = 110, minLineLength = 100, maxLineGap = 15)

   for zelenePrave in koordinate_zelene:
      for x1,y1,x2,y2 in zelenePrave:
        return int(x1), int(y1), int(x2), int(y2)
        