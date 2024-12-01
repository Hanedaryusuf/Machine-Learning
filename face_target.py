from deepface import DeepFace
#from deepface import VGGFace, Facenet, OpenFace, FbDeepFace
import pandas as pd

#model = VGGFace.loadModel()

df = DeepFace.find(img_path = "C:/test/img14.jpg", db_path = "C:/test", 
                   #model_name = 'VGGFace', model = model
                   )

#df.head(1)

print(df)
