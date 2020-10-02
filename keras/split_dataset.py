import splitfolders

SOURCE_DIR = './data/output/figures/mels'
OUT_DIR = './data/output/figures_split/mels'

splitfolders.ratio(SOURCE_DIR, output=OUT_DIR, seed=1337, ratio=(.8, .1, .1))