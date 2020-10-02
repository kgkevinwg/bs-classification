import os
from PIL import Image
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import argparse
import random
import tqdm
import shutil
import warnings
import cv2


from skimage import io
warnings.filterwarnings('ignore')
NOISE_DATA_ROOT = './data/xc/noises'
SIGNAL_DATA_ROOT = './data/xc/signal'
OUTPUT_DATA_ROOT ='./data/output'
SAMPLING_RATE = 44100
cm = plt.get_cmap('gnuplot')

parser = argparse.ArgumentParser(description="preprocess audio signal and noise to mels spectrogram and MFCC")
parser.add_argument('--n_noises_per_sample', type=int, default=20)
parser.add_argument('--n_data_per_class', type=int, default=3000)
args = parser.parse_args()

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled.astype(np.uint8)

def preprocess_figure(audio, out_shape):
    image = np.flip(scale_minmax(audio, 0, 255), axis=0)
    colored_image = cm(image)
    colored_image = np.uint8(colored_image * 255)
    colored_image = Image.fromarray(colored_image)
    colored_image = colored_image.resize(out_shape)
    return colored_image

def rotate_and_resize(im, out_shape , rotate=False):
    cv_image = np.array(im)
    #convert rgb to bgr
    # cv_image = cv_image[:, :, ::-1].copy()
    if rotate:
        cv_image = cv2.transpose(cv_image)
        cv_image = cv2.flip(cv_image, flipCode=1)
    cv_image = cv2.resize(cv_image, out_shape)
    return cv_image




def produce_histogram_MFCC(audio, sr):
    # x, sr = librosa.load(sample_audo_path, sr=44100)
    x_mels = librosa.feature.melspectrogram(audio, sr=sr)
    x_mels = librosa.power_to_db(x_mels)
    mfcc = librosa.feature.mfcc(audio, sr=sr)
    mels_image = preprocess_figure(x_mels, out_shape= (75, 128))
    # mels_image = rotate_and_resize(mels_image, (75, 128))
    mfcc_image = preprocess_figure(mfcc, out_shape= (75, 128))
    # mfcc_image = rotate_and_resize(mfcc_image, (75, 20))

    return mels_image, mfcc_image

def generate_filelist(data_root):
    filepath_list = list()
    for filename in os.listdir(data_root):
        filepath = os.path.join(NOISE_DATA_ROOT, filename)
        filepath_list.append(filepath)
    return filepath_list

def cut_audio(audio, start, stop):
    return audio[start:stop]

def get_audio_length(audio):
    audio_shape = audio.shape
    if len(audio_shape) == 1:
        return audio_shape[0]
    elif len(audio_shape) == 2:
        return audio_shape[1]

def combine_audio(audio1, audio2):
    return (audio1+audio2) / 2

def get_file_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == '__main__':
    noise_filelist = generate_filelist(NOISE_DATA_ROOT)

    for species_folder in tqdm.tqdm(os.listdir(SIGNAL_DATA_ROOT)):
        data_generated_on_class = 0
        while data_generated_on_class < args.n_data_per_class:
            for filename in tqdm.tqdm(os.listdir(os.path.join(SIGNAL_DATA_ROOT, species_folder))):
                if data_generated_on_class > args.n_data_per_class:
                    print("[INFO] breaking off-limit data per class")
                    break
                signal_filepath = os.path.join(SIGNAL_DATA_ROOT, species_folder, filename)
                signal_filename_base = get_file_basename(filename)
                print("[INFO] generating {} files for file {}".format(args.n_noises_per_sample, filename))
                signal_audio, sr = librosa.load(signal_filepath, sr=SAMPLING_RATE)
                signal_audio_length = get_audio_length(signal_audio)
                for i in range(args.n_noises_per_sample):
                    random_idx = random.randint(0, len(noise_filelist) - 1)
                    random_noise_filepath = noise_filelist[random_idx]
                    random_noise_filename_base = get_file_basename(random_noise_filepath)
                    random_noise, sr = librosa.load(random_noise_filepath, sr=SAMPLING_RATE)
                    random_noise_length = get_audio_length(random_noise)
                    random_noise_start = random.randint(0, random_noise_length - signal_audio_length)
                    random_noise_end = random_noise_start + signal_audio_length
                    random_noise_cut = cut_audio(random_noise, random_noise_start, random_noise_end)
                    combined_audio = combine_audio(signal_audio, random_noise_cut)
                    mels_image, mfcc_image = produce_histogram_MFCC(combined_audio, sr=SAMPLING_RATE)

                    save_dir = os.path.join(OUTPUT_DATA_ROOT, 'audio', species_folder, signal_filename_base)
                    figure_save_dir = os.path.join(OUTPUT_DATA_ROOT, 'audio', species_folder, signal_filename_base, 'figures')
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    if not os.path.isdir(figure_save_dir):
                        os.makedirs(figure_save_dir)

                    combined_save_filepath = os.path.join(save_dir,"mixed_"+signal_filename_base+"_"+\
                                                          random_noise_filename_base+".wav")
                    signal_save_filepath = os.path.join(save_dir, signal_filename_base+".wav")
                    noise_save_filepath = os.path.join(save_dir, "cut_{}_{}_".format(random_noise_start, random_noise_end)+\
                                                       random_noise_filename_base+".wav")
                    mfcc_save_filepath = os.path.join(figure_save_dir, "mfcc_{}_{}-{}-{}.png".format(signal_filename_base, random_noise_filename_base, random_noise_start, random_noise_end))
                    mels_save_filepath = os.path.join(figure_save_dir, "mels_{}_{}-{}-{}.png".format(signal_filename_base, random_noise_filename_base, random_noise_start, random_noise_end))


                    librosa.output.write_wav(combined_save_filepath, combined_audio, sr)
                    shutil.copy(signal_filepath, signal_save_filepath)
                    shutil.copy(random_noise_filepath, noise_save_filepath)
                    # io.imsave(mfcc_save_filepath, mfcc_image)
                    # io.imsave(mels_save_filepath, mels_image)
                    mels_image.save(mels_save_filepath)
                    mfcc_image.save(mfcc_save_filepath)
                    # cv2.imwrite(mels_save_filepath, mels_image)
                    # cv2.imwrite(mfcc_save_filepath, mfcc_image)
                    if data_generated_on_class % 500 == 0:
                        print('[INFO] generated {} datas for class {}'.format(data_generated_on_class, species_folder))
                    data_generated_on_class += 1














