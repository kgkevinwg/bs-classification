from model import MainModel
import tensorflow.keras as keras
import os
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



MELS_DATA_ROOT = './data/output/figures_split/mels'
MFCC_DATA_ROOT = './data/output/figures_split/mfcc'
BATCH_SIZE = 4

def generate_multiple_generator(generator, dir):
    gen_mels = generator.flow_from_directory(os.path.join(MELS_DATA_ROOT, dir), shuffle=False, batch_size=BATCH_SIZE)
    gen_mfcc = generator.flow_from_directory(os.path.join(MFCC_DATA_ROOT, dir), shuffle=False, batch_size=BATCH_SIZE)
    print(len(list(gen_mels)))

    while True:
        x_mels = gen_mels.next()
        x_mfcc = gen_mfcc.next()
        yield [x_mels[0], x_mfcc[0]], x_mels[1]

datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = generate_multiple_generator(datagen, 'train')
val_generator = generate_multiple_generator(datagen, 'val')
test_generator = generate_multiple_generator(datagen, 'test')

mc_callback = keras.callbacks.ModelCheckpoint('checkpoints', verbose=1)

model = MainModel()
model.build_model()
history = model.final_model.fit_generator(train_generator, epochs=100, validation_data=val_generator, \
                                          callbacks=[mc_callback], \
                                          steps_per_epoch=10400 // BATCH_SIZE, \
                                          validation_steps=1300 // BATCH_SIZE, verbose=1)






