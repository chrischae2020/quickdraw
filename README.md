# BrunonianQuickDraw

To run model:

 - download quickdraw image files (.npy) from here: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
 - place files in directory /data/
 - run model using python command: python3 -u code/main.py --num_imgs=10000 --epochs=10 --latent_size=500 --batch_size=64 --save_outputs=True
 - output files will be available in /output/ by default

Best run: 10000 images per category, 500 latent size, 64 batch size for test accuracy of 86.7%, beating our initial goal of 85%


After running the model...
To Play with GUI:
  
 - run using command: python3 ./code/gui.py

You can choose any of the categories on the left to draw

Have fun!!
