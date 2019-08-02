# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:21:53 2019

@author: Prathima
"""
#import required modules
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, Dense, add, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from keras.layers import Input
import tkinter
from tkinter import filedialog
import os

token_dir = r"..\Flicker8k_Dataset\ned\Flickr8k.token.txt"
image_captions = open(token_dir).read().split('\n')
caption = {}    
for i in range(len(image_captions)-1):
    id_capt = image_captions[i].split("\t")
    id_capt[0] = id_capt[0][:len(id_capt[0])-2] # to rip off the #0,#1,#2,#3,#4 from the tokens file
    if id_capt[0] in caption:
        caption[id_capt[0]].append(id_capt[1])
    else:
        caption[id_capt[0]] = [id_capt[1]]
train_imgs_id = open(r"..\Flicker8k_Dataset\ned\Flickr_8k.trainImages.txt").read().split('\n')[:-1]
train_imgs_captions = open("trainimgs.txt",'w')         #create a file with training image ids
for img_id in train_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        train_imgs_captions.write(img_id+"\t"+desc+"\n")
        train_imgs_captions.flush()
train_imgs_captions.close()

test_imgs_id = open(r"..\Flicker8k_Dataset\ned\Flickr_8k.testImages.txt","w").readlines()
test_imgs_captions = open("testimgs.txt",'w')           #create a file with training image ids
for img_id in test_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        test_imgs_captions.write(img_id+"\t"+desc+"\n")
        test_imgs_captions.flush()
test_imgs_captions.close()

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')
new_input = model.input
new_output = model.layers[-2].output
model_new = Model(new_input, new_output)

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

images = r"..\Flicker8k_Dataset\flickr8k"
train_imgs_id = open(r"..\Flicker8k_Dataset\ned\Flickr_8k.trainImages.txt").read().split('\n')[:-1]
encoding_train = {}
for img in tqdm(train_imgs_id): #tqdm instantly make your loops show a smart progress meter
    path = images+"\\" + str(img)
    encoding_train[img] = encode(path)
with open("encoded_train_images_inceptionV3.p", "wb") as encoded_pickle: 
    pickle.dump(encoding_train, encoded_pickle) #python object can be pickled so that it can be saved on disk.
encoding_train = pickle.load(open('encoded_train_images_inceptionV3.p', 'rb'))

#for testing your own images,in the below part just change the 'test_images_id' with the file having the image_ids of your images
#the code for that is in the end
test_imgs_id = open(r"..\Flicker8k_Dataset\ned\Flickr_8k.testImages.txt").read().split('\n')[:-1]
encoding_test = {}
for img in tqdm(test_imgs_id):
    img=img.split("\n")[0]
    print(img)
    encoding_test[img] = encode(img)
with open("encoded_test_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)
encoding_test = pickle.load(open('encoded_test_images_inceptionV3.p', 'rb'))

dataframe = pd.read_csv(r'C:\ML\inception\trainimgs.txt', delimiter='\t')
captionz = []
img_id = []
dataframe = dataframe.sample(frac=1)
iter = dataframe.iterrows()

for i in range(len(dataframe)):
    nextiter = next(iter)
    captionz.append(nextiter[1][1])
    img_id.append(nextiter[1][0])

no_samples=0
tokens = []
tokens = [i.split() for i in captionz]
for caption in captionz:
    no_samples+=len(caption.split())-1

vocab= pickle.load(open('vocab.p', 'rb'))          #pretrained weights for vocabulary
print(len(vocab))
vocab_size = len(vocab)
word_idx = {val:index for index, val in enumerate(vocab)}
idx_word = {index:val for index, val in enumerate(vocab)}
print(word_idx['end'])
caption_length = [len(caption.split()) for caption in captionz]
max_length = max(caption_length)
print(max_length)                       # maximum lenght of a caption.

def data_process(batch_size):
    partial_captions = []
    next_words = []
    images = []
    total_count = 0
    while 1:
    
        for image_counter, caption in enumerate(captionz):
            current_image = encoding_train[img_id[image_counter]]
    
            for i in range(len(caption.split())-1):
                total_count+=1
                partial = [word_idx[txt] for txt in caption.split()[:i+1]]
                partial_captions.append(partial)
                next = np.zeros(vocab_size)
                next[word_idx[caption.split()[i+1]]] = 1
                next_words.append(next)
                images.append(current_image)

                if total_count>=batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_length, padding='post')
                    total_count = 0
                
                    yield [[images, partial_captions], next_words]
                    partial_captions = []
                    next_words = []
                    images = []

EMBEDDING_DIM = 300
# image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# partial caption sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
fin_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
fin_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])      #compile the model
epoch = 5
batch_size = 128
fin_model.fit_generator(data_process(batch_size=batch_size), steps_per_epoch=no_samples/batch_size, epochs=epoch, verbose=1, callbacks=None)   #start training
fin_model.save("Weights_1")           #save the weights


#testing phase
fin_model = load_model('Weights_1')

def beam_search_predictions(image_file, beam_index = 3):
  
    start = [word_idx["<start>"]]  
    start_word = [[start, 0.0]]  
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            now_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            e = encoding_test[image_file]
            preds = fin_model.predict([np.array([e]), np.array(now_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx_word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


image_file="3255482333_5bcee79f7e.jpg"      #an image from the test dataset
test_image =  images + "\\" + image_file

print ('Beam Search, k=3:', beam_search_predictions(image_file, beam_index=1))
print ('Beam Search, k=5:', beam_search_predictions(image_file, beam_index=3))
print ('Beam Search, k=9:', beam_search_predictions(image_file, beam_index=5))


#frontend(only for testing with your images)
root = tkinter.Tk()
root.lift()
root.withdraw()           #use to hide tkinter window
root.focus_force() 

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askopenfilename(initialdir=currdir, title='Please select a directory',filetypes = (("jpeg files","*.jpg"),("Portable Network Graphics","*.png"),("all files","*.*")))
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir
  
test_image = search_for_file_path()         #test_image is the test image file path
word=test_image.split('/')[-1]
print ("\nfile_path_variable=", word.split("\n")[0])
fp=open("test_image_ids.txt",'a') 
fp.write(word+"\n")
fp.close()

image_file = word   
#after this continue with the above code for testing.
