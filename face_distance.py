import os
import sys

tool = './bin/face_embedding'
pic1 = sys.argv[1]
pic2 = sys.argv[2]
command = tool + ' ' + pic1 + ' ' + pic2
out = os.popen(command)
info = out.readlines()
dist = float(info[0])
print(dist)
