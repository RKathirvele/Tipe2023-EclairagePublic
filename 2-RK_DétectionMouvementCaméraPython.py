import cv2
import numpy as np


capture = cv2.VideoCapture(1) # démarrer la caméra

# 300 image, seuil de comparaison de la position de chaque pixel, détécte les ombres)
fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)  

nbimage = 0 

while(True): #Tant que la caméra est allumé 
	ret, frame = capture.read() # retourne vrai ou faux et prend image par image

	nbimage += 1

	#(0,0) est la position , fx et fy sont la largeur et hauteur de l'image
	nouvelleformeimage = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50) 

        #  Obtenir le premier plan pour donner une image en noir et blanc  
	fgmask = fgbg.apply(nouvelleformeimage)
	
	nbpixel = np.count_nonzero(fgmask) # compter le nombre de pixel non nul

	print('Frame: %d, Pixel Count: %d' % (nbimage, nbpixel))

	
	if (nbimage > 1 and nbpixel > 5000): # 5000 nombre de pixel où il y'a un changement 
		print('Mouvement detecte')
		cv2.putText(nouvelleformeimage, 'mouvementdetecte', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Image', nouvelleformeimage)
	cv2.imshow('Pixel', fgmask)


	k = cv2.waitKey(1) & 0xff #fermer la caméra avec la touche échap
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()

