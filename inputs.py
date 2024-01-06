from PIL import Image
import numpy as np


def image_to_np():
    try:
        img = Image.open("input.png").convert("L")
    except:
        print("Could not open image")
    img = img.resize((28,28))
    # x, y = img.size
    # size = max(280, x, y)
    # new = Image.new('RGBA', (size, size), (255,255,255,0))
    # new.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    ni = np.asarray(img)
    ni = ni.reshape((28,28,1))
    ni = ni / 255.0
    ni = np.where(ni==1., 0., ni)
    return ni
# if os.path.isfile(file_name):
# with open(file_name, 'rb') as datafile:
#     save_object = pickle.load(datafile)