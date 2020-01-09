# Part 1 - Building the CNN
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#input img -> into 2D array ->apply feature detector -> feature map
#nb_filter = number of filter we choose the number of feature map will be created
#no of rows and no of col in feature detector
#Convolution2D(nb_filter, row, col, border_mode)
#input shape ->the input shape on which apply feature detector
#input_shape(pix,pix,no of channel) #tenforflow seq
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#pool_sizerow,col() the size of the stride that we pull in the img
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer -> reduce overfitting and increase accuracy 
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#hidden nodes -> not a too small number arount 100 is a good choice
#units = output_dim
classifier.add(Dense(output_dim = 128, activation = 'relu')) #hidden layer
classifier.add(Dense(units = 1, activation="sigmoid")) #output layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
#pre-process img to prevent over-fitting using img augmentation process 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, #pixel scale from 0-255 to 0-1
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True) #img will be flipped

test_datagen = ImageDataGenerator(rescale=1./255) #pixel scale from 0-255 to 0-1

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64), #size of img that is expected in cnn model. Here 64x64
                                                 batch_size=32, #after which the weights will be updated
                                                 class_mode='binary') #if DV are binary or more then

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000, #no of img in the training here we have 8000 img
                         epochs=25,
                         validation_data=test_set, #test_set 
                         validation_steps=2000) #no of img in test set