import json
import numpy as np
from pprint import pprint

with open('output.json') as f:
    data = json.load(f)

#pprint(data)

# for d in data:
#     #print(d[''])
#     for f in d['clips']:
#         print(f.get('000'))



for j in range(len(data)):

    for i in data[j]["clips"]:
        print (i["features"])




print (len(data))  # total number of videos
print (len(data[0]["clips"]))  # total number of features from a video
dd = data[0]

ddd = data[0]["clips"]

dddd = data[0]["clips"][0]

ddddd = data[0]["clips"][0]["features"]


np_a = np.asarray(ddddd)


print('end')