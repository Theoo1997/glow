# Generates y_labels (ground truth class labels)

# This script must be run AFTER the raw images have been processed and saved as .bin

# To save as .bin, use a script like:
# https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/buildPersonDetectionDatabase.py
# but save as .bin instead of .jpg

import os
from PIL import Image

GROUND_TRUTH_FILENAME='y_labels.txt'
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

PERSON_DIR =  dir_path + '/vw_coco2014_96/person/'
NON_PERSON_DIR = dir_path + '/vw_coco2014_96/non_person/'
def generate_file_contents():
  outfile = open(GROUND_TRUTH_FILENAME, "a")
  counter=0
  for file in os.listdir(PERSON_DIR):
    input_filename = os.fsdecode(file)
    if input_filename[5]=='v':
      line = '%s %d \n'%(file[:-3] + "png",1)
      outfile.write(line)
      img = Image.open(PERSON_DIR + file)
      png_image = dir_path + "/images/" + file[:-3] + "png"
      img.save(png_image, format='PNG')

  for file in os.listdir(NON_PERSON_DIR):
      input_filename = os.fsdecode(file)
      if input_filename[5]=='v':
        line = '%s %d \n'%(file[:-3] + "png",0)
        outfile.write(line)
        img = Image.open(NON_PERSON_DIR + file)
        png_image = dir_path + "/images/" + file[:-3] + "png"
        img.save(png_image, format='PNG')

  outfile.flush()
  outfile.close()

generate_file_contents()
