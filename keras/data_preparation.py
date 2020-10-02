import shutil
import os
from tqdm import tqdm

PREPROCESS_OUTPUT_ROOT = './data/output/audio'
MELS_DESTINATION_ROOT = './data/output/figures/mels'
MFCC_DESTINATION_ROOT = './data/output/figures/mfcc'


for species in tqdm(os.listdir(PREPROCESS_OUTPUT_ROOT)):
    species_dir = os.path.join(PREPROCESS_OUTPUT_ROOT, species)
    for file in os.listdir(species_dir):
        figure_dir = os.path.join(species_dir, file, 'figures')
        for figure in os.listdir(figure_dir):
            figure_file_path = os.path.join(figure_dir, figure)
            if figure.startswith('mels'):
                destination_dir = os.path.join(MELS_DESTINATION_ROOT, species)
            else:
                destination_dir = os.path.join(MFCC_DESTINATION_ROOT, species)

            destination_file_path = os.path.join(destination_dir, figure)
            if not os.path.isdir(destination_dir):
                os.makedirs(destination_dir)
            # print(figure_file_path)
            # print(destination_file_path)
            # print("="*10)

            shutil.copy(figure_file_path, destination_file_path)