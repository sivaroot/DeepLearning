import numpy as np
import os
from scipy import  misc
from keras.models import model_from_json
import pickle
import matplotlib.pyplot as plt



classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Model is now loaded in the disk")
w=64
h=64
ax = []
img=os.listdir("../dataset/testing")
fig=plt.figure(figsize=(8, 8))
columns = 8
rows = int(len(img)/8) + 1
print('Row : ', rows)
i=1
for image_label in img:
    image=np.array(misc.imread("../dataset/testing/"+image_label))
    image = misc.imresize(image, (64, 64))
    tmp = image
    
    
    
    image = np.array([image])
    image = image.astype('float32')
    image = image / 255.0
    prediction=loaded_model.predict(image)
    print(prediction)
    print(np.max(prediction))
    print(int_to_word_out[np.argmax(prediction)])
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title(str(int_to_word_out[np.argmax(prediction)]))
    ax[-1].set_xlabel(str(int(np.max(prediction)*100)) +"%")
    i=i+1
    plt.imshow(tmp)
plt.show()
