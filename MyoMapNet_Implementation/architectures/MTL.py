from keras.layers import Input, Flatten, Dense, Multiply, Lambda
from keras.layers import Conv2D, concatenate,  MaxPooling2D, UpSampling2D, Dropout, Conv3D, MaxPooling3D, UpSampling3D
from keras.optimizers import Adam
from keras import backend as K
from keras.models import  Model


conv_size = (3, 3)
pool_size = (2, 2)
smooth = 1.
filter_size = (3,3)
pool_size = (2,2)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def func(x):

    greater = K.greater_equal(x, 0.5) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1

    return greater

def multiaskunet2D(input_size=(512,512,1)):
    #Input for the mulitask
    inputs = Input(input_size)

    #The encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #Decoder for segmentation

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation="sigmoid", name="Segmentation")(conv9)

    conv10_binarized = K.greater_equal(conv10, 0.5)  # will return boolean values
    conv10_binarized = K.cast(conv10_binarized, dtype=K.floatx())
    mul1 = Multiply()([conv10_binarized, inputs])

    #Decoder for reconstruction
    up11 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge11 = concatenate([drop4, up11], axis=3)
    conv11 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv11 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    up12 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv11))
    merge12 = concatenate([conv3, up12], axis=3)
    conv12 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)

    up13 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv12))
    merge13 = concatenate([conv2, up13], axis=3)
    conv13 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv13 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    up14 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv13))
    merge14 = concatenate([conv1, up14], axis=3)
    conv14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
    conv14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    conv14 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)

    conv15 = Conv2D(1, 1, activation="sigmoid", name="Reconstruction")(conv14)


    #Pathology classification
    dense1 = Flatten()(drop5)
    dense1 = Dense(128, activation="elu")(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(1, activation="sigmoid", name="classification")(dense1)

    #Radiomics
    conv16 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mul1)
    conv16 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv16)
    pool16 = MaxPooling2D(pool_size=(2, 2))(conv16)
    conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool16)
    conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    pool17 = MaxPooling2D(pool_size=(2, 2))(conv17)

    dense2 = Flatten()(pool17)
    dense2 = Dense(128, activation="elu")(dense2)
    dense2 = Dropout(0.5)(dense2)
    dense2 = Dense(1, activation="sigmoid", name="Radiomics")(dense2)

    model = Model(input=inputs, output=[conv10, conv15, dense1, dense2])

    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss, "mean_squared_error", "binary_crossentropy", "binary_crossentropy"],
                  metrics=[dice_coef, "accuracy", "accuracy", "accuracy"])

    model.summary()

    return model



def multiaskunet3D(input_size=(48,48,48,1)):
    #Input for the mulitask
    inputs = Input(input_size)

    #The encoder
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #Decoder for segmentation

    up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv3D(1, 1, activation="sigmoid", name="segmentation")(conv9)

   # conv10_binarized = K.greater_equal(conv10, 0.5)  # will return boolean values
    #conv10_binarized = K.cast(conv10_binarized, dtype=K.floatx())
    mul1 = Lambda(func)
    mul1 = mul1(conv10)
    mul1 = Multiply()([mul1, inputs])

    #Decoder for reconstruction
    up11 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge11 = concatenate([drop4, up11], axis=-1)
    conv11 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv11 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    up12 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv11))
    merge12 = concatenate([conv3, up12], axis=-1)
    conv12 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)

    up13 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv12))
    merge13 = concatenate([conv2, up13], axis=-1)
    conv13 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv13 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    up14 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv13))
    merge14 = concatenate([conv1, up14], axis=-1)
    conv14 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
    conv14 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    conv14 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)

    conv15 = Conv3D(1, 1, activation="sigmoid", name="reconstruction")(conv14)


    #Pathology classification
    dense1 = Flatten()(drop5)
    dense1 = Dense(128, activation="elu")(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(1, activation="sigmoid", name="classification")(dense1)

    #Radiomics
    conv16 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mul1)
    conv16 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv16)
    pool16 = MaxPooling3D(pool_size=(2, 2, 2))(conv16)
    conv17 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool16)
    conv17 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    pool17 = MaxPooling3D(pool_size=(2, 2, 2))(conv17)

    dense2 = Flatten()(pool17)
    dense2 = Dense(128, activation="elu")(dense2)
    dense2 = Dropout(0.5)(dense2)
    dense2 = Dense(1, activation="sigmoid", name="radiomics")(dense2)

    model = Model(input=inputs, output=[conv10, conv15, dense1, dense2])

    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss, "mean_squared_error", "binary_crossentropy", "binary_crossentropy"],
                  metrics=[dice_coef, "accuracy", "accuracy", "accuracy"])

    model.summary()

    return model

def multiaskunet3D_radiomics_seg_encodeur(input_size=(48,48,48,1)):
    #Input for the mulitask
    inputs = Input(input_size)

    #The encoder
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    #Decoder for segmentation

    up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv3D(1, 1, activation="sigmoid", name="segmentation")(conv9)

   # conv10_binarized = K.greater_equal(conv10, 0.5)  # will return boolean values
    #conv10_binarized = K.cast(conv10_binarized, dtype=K.floatx())
    mul1 = Lambda(func)
    mul1 = mul1(conv10)
    mul1 = Multiply()([mul1, inputs])

    #Decoder for reconstruction
    up11 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge11 = concatenate([drop4, up11], axis=-1)
    conv11 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv11 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    up12 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv11))
    merge12 = concatenate([conv3, up12], axis=-1)
    conv12 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)

    up13 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv12))
    merge13 = concatenate([conv2, up13], axis=-1)
    conv13 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv13 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    up14 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv13))
    merge14 = concatenate([conv1, up14], axis=-1)
    conv14 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
    conv14 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    conv14 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)

    conv15 = Conv3D(1, 1, activation="sigmoid", name="reconstruction")(conv14)


    #Pathology classification
    dense1 = Flatten()(drop5)
    dense1 = Dense(128, activation="elu")(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(1, activation="sigmoid", name="classification")(dense1)

    #Radiomics
    conv16 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mul1)
    conv16 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv16)
    pool16 = MaxPooling3D(pool_size=(2, 2, 2))(conv16)
    conv17 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool16)
    conv17 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    pool17 = MaxPooling3D(pool_size=(2, 2, 2))(conv17)

    dense2 = Flatten()(pool17)
    dense_encodeur = Flatten()(drop5)
    dense2 = concatenate([dense2, dense_encodeur])
    dense2 = Dense(128, activation="elu")(dense2)
    dense2 = Dropout(0.5)(dense2)
    dense2 = Dense(1, activation="sigmoid", name="radiomics")(dense2)

    model = Model(input=inputs, output=[conv10, conv15, dense1, dense2])

    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss, "mean_squared_error", "binary_crossentropy", "binary_crossentropy"],
                  metrics=[dice_coef, "accuracy", "accuracy", "accuracy"])

    model.summary()

    return model

def CNN_unet2D(input_size=(512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    dense = Flatten()(pool2)
    dense = Dense(128, activation="elu")(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(1, activation="sigmoid")(dense)

    return dense



