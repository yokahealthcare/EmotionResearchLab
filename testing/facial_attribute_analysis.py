import json
import sys

from deepface import DeepFace

pic1 = "../asset/img/aerith#2.jpg"
pic2 = "../asset/img/aerith#4.jpg"

objs = DeepFace.analyze(img_path=pic1, actions=['age', 'gender', 'race', 'emotion'])
# Convert and write JSON object to file
with open(f"{sys.argv[0]}.json", "w") as outfile:
    json.dump(objs, outfile)
