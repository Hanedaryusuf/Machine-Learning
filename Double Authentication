from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


img1_path = "C:/test/test1.jpg"
img2_path = "C:/test/test2.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

plt.imshow(img1[:,:,::-1])
plt.show()

plt.imshow(img2[:,:,::-1])
plt.show()

resp = DeepFace.verify(img1_path, img2_path
                       # , model_name = "DeepFace"
                       #,distance_metric = "euclidean_l2"
                       )

print(resp["verified"])
