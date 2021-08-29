import glob
import os
import random
import shutil
from pathlib import Path


def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


def prepare_img_dir(original_dir, source_dir, task_number):

    try:
        shutil.rmtree(original_dir)
    except:
        pass

    p = Path(source_dir)  # os-agnostic
    images_files =[]
    if p.is_dir():  # dir
        images_files = glob.glob(str(p / '**' / '*.jpg'), recursive=True)

    image_list = [i for i in range(1, len(images_files))]
    random.shuffle(image_list)

    dirs = partition(image_list, task_number)

    Path(original_dir).mkdir(parents=True, exist_ok=True)

    inx = 0
    for images in dirs:
        Path(original_dir + "/" + str(inx).zfill(3)).mkdir(parents=True, exist_ok=True)
        #print(images)
        jnx = 0
        for img in images:
            #print(os.path.join(source_dir, str(img).zfill(5) + ".jpg"))
            #print(os.path.join(original_dir, str(inx).zfill(3), str(jnx).zfill(5) + ".jpg"))
            shutil.copy2(os.path.join(source_dir, str(img).zfill(5) + ".jpg"),
                         os.path.join(original_dir,  str(inx).zfill(3), str(jnx).zfill(5) + ".jpg"))


            jnx += 1
        inx += 1