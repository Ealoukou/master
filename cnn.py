
# coding: utf-8

# In[1]:


import pandas as pd 
from io import StringIO

# We import our data
df_train = pd.read_csv('Downloads/master/Content Analytics for Business Analysts/amazon data/train_v2.csv')
df_train.head()


# In[2]:


df_train.describe()


# In[3]:


# We observe that the data have multible labels, so we split them in order to count the most frequent categories
all_tags = [item for sublist in list(df_train['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
print('total of {} non-unique tags in all training images'.format(len(all_tags)))
print('average number of labels per image {}'.format(1.0*len(all_tags)/df_train.shape[0]))


# In[4]:


# the frequency table for top 10 labels
tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
tags_counted_and_sorted.head(10)


# In[5]:


# plot with the most frequent labels
tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))


# In[6]:


# We import some images in order to see different labels and how they represented in images
import cv2
import matplotlib
import matplotlib.pyplot as plt

new_style = {'grid': True}
plt.rc('axes', **new_style)
_, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(20, 20))
i = 0
for f, l in df_train[:16].values:
    img = cv2.imread('Downloads/master/Content Analytics for Business Analysts/amazon data/train-jpg/{}.jpg'.format(f))
    ax[i // 4, i % 4].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[i // 4, i % 4].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    i += 1
    
plt.show()


# In[7]:


# We define the path of our data and we choose to proceed with only two different categories 
#that are the most frequent and important for our analysis. We choose agriculture and road category.
import os
train_path = 'Downloads/master/Content Analytics for Business Analysts/amazon data/train-jpg/' 
train = [train_path+f for f in os.listdir(train_path) if f.endswith('.jpg')]
print ('loaded', len(train)-1, 'train examples')


train_examples = []
for f, l in df_train.values:
    #print(l)
    fpath = 'Downloads/master/Content Analytics for Business Analysts/amazon data/train-jpg/{}.jpg'.format(f)
    if "agriculture" in l and 'road'not in l:
        train_examples.append((fpath, 0))
    elif "road" in l and 'agriculture'not in l:
        train_examples.append((fpath, 1))

print(len(train_examples))


# In[8]:


# We use skimage algorithm in order to rescale the images from (256,256,3) to (128,128,1). 
# We also choose to gray scale the images in order to gain time and memory.
import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread

def examples_to_dataset(examples, block_size=2):
    X = []
    y = []
    for train_path, label in examples:
        img_train = imread(train_path, as_grey=True) 
        #print( img_train.shape)
        img_train = block_reduce(img_train, block_size=(block_size, block_size), func=np.mean)
        #print( img_train.shape)
        X.append(img_train)
        y.append(label)
    return np.expand_dims(np.asarray(X),-1), np.asarray(y)

get_ipython().run_line_magic('time', 'X, y = examples_to_dataset(train_examples)')


# In[9]:


X = X.astype(np.float32) / 255. # we normalize the data 
y = y.astype(np.int32)
print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)


# In[10]:


import numpy as np
import math
from io import BytesIO
import numpy as np
import PIL.Image
import IPython.display
import shutil

def find_rectangle(n, max_ratio=2):
    sides = []
    square = int(math.sqrt(n))
    for w in range(square, max_ratio * square):
        h = n / w
        used = w * h
        leftover = n - used
        sides.append((leftover, (w, h)))
    return sorted(sides)[0][1]

def make_mosaic(images, n=None, nx=None, ny=None, w=None, h=None):
    if n is None and nx is None and ny is None:
        nx, ny = find_rectangle(len(images))
    else:
        nx = n if nx is None else nx
        ny = n if ny is None else ny
    images = np.array(images)
    if images.ndim == 2:
        side = int(np.sqrt(len(images[0])))
        h = side if h is None else h
        w = side if w is None else w
        images = images.reshape(-1, h, w)
    else:
        h = images.shape[1]
        w = images.shape[2]
    image_gen = iter(images)
    mosaic = np.empty((h*ny, w*nx))
    for i in range(ny):
        ia = (i)*h
        ib = (i+1)*h
        for j in range(nx):
            ja = j*w
            jb = (j+1)*w
            mosaic[ia:ib, ja:jb] = next(image_gen)
    return mosaic

def show_array(a, fmt='jpeg', filename=None):
    a = np.squeeze(a)
    a = np.uint8(np.clip(a, 0, 255))
    image_data = BytesIO()
    PIL.Image.fromarray(a).save(image_data, fmt)
    if filename is None:
        IPython.display.display(IPython.display.Image(data=image_data.getvalue()))
    else:
        with open(filename, 'w') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)


# In[12]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils


# In[13]:


# convert classes to vector
nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

# shuffle all the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# prepare weighting for classes since they're unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)


# In[14]:


from keras.models import Sequential
from keras.layers import Dense

nb_filters = 64
nb_pool = 2
nb_conv = 3


model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[15]:


validation_split = 0.10
model.fit(X, y, batch_size=128, 
          class_weight=class_weight, epochs=5, 
          verbose=1, 
          validation_split=validation_split)


# In[16]:


open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.plot(model.model.history.history['loss'])
plt.plot(model.model.history.history['acc'])
plt.plot(model.model.history.history['val_loss'])
plt.plot(model.model.history.history['val_acc'])
plt.show()


# In[18]:


from sklearn.metrics import roc_auc_score
n_validation = int(len(X) * validation_split)
y_predicted = model.predict(X[-n_validation:])
print (roc_auc_score(y[-n_validation:], y_predicted))


# In[19]:


# import keras
# from keras.models import model_from_json
# model = model_from_json(open('model.json').read())
# model.load_weights('weights.h5')

