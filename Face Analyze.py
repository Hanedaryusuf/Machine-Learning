from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img_path = "C:/test/test1.jpg"

img = cv2.imread(img_path)

#Image show
plt.imshow(img[:,:,::-1])
plt.show()

resp = DeepFace.analyze(img_path, 
                       #For special parts
                       # ['age', 'emotion']
                        )

#For important things(But they can't work)
 #print(resp['age'])
 #print(resp["gender"])
 #print(resp['dominant_emotion'])
 #print(resp["dominant_race"])


print(resp) 
