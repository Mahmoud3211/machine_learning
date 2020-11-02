import os
from pp_utils import wall_thickness_detector
from PIL import Image

empty_image = Image.open(os.path.join(os.getcwd(), 'test', 'a2.bmp'))

wall_thickness_image, messages, count = wall_thickness_detector(empty_image).detect()
for message in messages:
    print(message)
print(f'number of flawed pipes : {count}')
wall_thickness_image.save('seg.jpg')