import glob
import os 
from shutil import copyfile

images = glob.glob('D:\Work\Wakeb\power_pipes\X-ray\power pipes pic\*\*\*.bmp')
print('original:', len(images))

file_path = 'D:/Work/Wakeb/power_pipes/X-ray/all_images/*.bmp'
copied = glob.glob(file_path)
print('copied:', len(copied))
images = [x.split('\\')[-1] for x in images]
copied = [x.split('\\')[-1] for x in copied]
# images.append('ay haga')
# print(copied)
print('Difference:',list(set(images) - set(copied)))
print('Difference:',list(set(copied) - set(images)))
print(len(set(images)))
print(len(set(copied)))

import pandas as pd

df = pd.DataFrame(images)
# print(df.head())
print(df[df.duplicated()])


# for i, image in enumerate(images):
#     copyfile(image, os.path.join(file_path, image.split('\\')[-1]))    
#     print(str({i+1}), '--', image.split("\\")[-1], 'image has been copied')