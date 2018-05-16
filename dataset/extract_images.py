import base64
import csv
import os
import argparse

parser = argparse.ArgumentParser(description='Extract images from csv file')
parser.add_argument('--tsv-name', type=str, default='')
parser.add_argument('--out-dir', type=str, default='MS-Celeb-1M')
args = parser.parse_args()


filename = args.tsv_name
outputDir = args.out_dir
assert filename != '', "No tsv file!"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
i = 0
with open(filename, 'r') as tsvF:
    reader = csv.reader(tsvF, delimiter='\t')
    
    for row in reader:
      MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

      saveDir = os.path.join(outputDir, MID)
      savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

      if not os.path.exists(saveDir):
        os.mkdir(saveDir)
      with open(savePath, 'wb') as f:
        f.write(data)

      i += 1

print("Extracted {} images.".format(i))
