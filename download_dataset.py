import os
import csv
import requests
from PIL import Image
import cv2
import numpy as np
import os
import csv
import requests
from PIL import Image
#from keras.utils.np_utils import to_categoricals
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


#***************************Image downloading**********************************************
def download_images_from_csv(csv_file):
    with open(csv_file, 'r',encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists
        for idx, row in enumerate(reader, start=1):
            url = row[7]  # Assuming the URL is in the third column
            name = row[0]
            
            # Create folder if it doesn't exist
            folder_name = f'./img/{idx}'
            os.makedirs(folder_name, exist_ok=True)
            print(name,'@@@@@@@@@@@@@@@@@@@')
            # Download image
            try:
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Save the image using Pillow

                    for i in range(1,7):
                        image_name = f'{name}_{i}.jpg'
                        image_path = os.path.join(folder_name, image_name)
                        with open(image_path, 'wb') as img_file:
                            img_file.write(response.content)
                    print(f"Image {idx} downloaded successfully.")
                    
                    # Open the image using Pillow to verify
                    #img = Image.open(image_path)
                    #img.show()  # This will open the image with the default image viewer
                else:
                    print(f"Failed to download image {idx}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading image {idx}: {str(e)}")

# Usage
csv_file_path = './indian dataset_123.csv'  # Path to your CSV file
download_images_from_csv(csv_file_path)


#*****************************Image features***************************************

path = "./img"#'./images'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        name = int(name)
        if name not in labels:
            labels.append(name)
labels.sort()            
print(labels,'##############################')

for i in range(len(labels)):
    label = labels[i]
    arr = os.listdir(path+'/'+str(label))
    for j in range(len(arr)):
        img = cv2.imread(path+'/'+str(label)+"/"+str(arr[j]))
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(64,64,3)
        X_train.append(im2arr)
        Y_train.append(label)
        #print(path+'/'+str(label)+"/"+str(arr[j]))
        

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)

X_train = X_train.astype('float32')
X_train = X_train/255
    
test = X_train[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)
np.save('./model/X.txt',X_train)
np.save('./model/Y.txt',Y_train)
print('done')







