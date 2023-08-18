import cv2
import face_recognition as fr

imgOperador = fr.load_image_file("dataset/" + "Operador.png")
imgOperador = cv2.cvtColor(imgOperador,cv2.COLOR_BGR2RGB)

imgComparador = fr.load_image_file("dataset/" + "Comparador.png")
imgComparador = cv2.cvtColor(imgComparador,cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgOperador)[0]
cv2.rectangle(imgOperador,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

encondeOperador = fr.face_encodings(imgOperador)[0]
encodeComparador = fr.face_encodings(imgComparador)[0]

comparacao = fr.compare_faces([encondeOperador],encodeComparador)
coordenadas = fr.face_distance([encondeOperador],encodeComparador)

print('\nResultados:')
print(comparacao,coordenadas)
cv2.imshow('Operador',imgOperador)
cv2.imshow('Comparador',imgComparador)
cv2.waitKey(0)
