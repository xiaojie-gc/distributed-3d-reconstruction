import base64
import multiprocessing
import time
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
import numpy as np


def write(fg_vertex):
    s = ""
    for i in range(len(fg_vertex)):
        for j in range(len(fg_vertex[i])):
            s += str(fg_vertex[i][j]) + ' '
            # f.write(str(fg_vertex[i][j]))
            # f.write(' ')
        s += '\n'
    return s


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def merge(fg, bg, result):
    start = time.time()

    fg_ply = PlyData.read(fg)
    bg_ply = PlyData.read(bg)

    fg_vertex = fg_ply.elements[0].data
    # fg_face = fg_ply.elements[1].data

    bg_vertex = bg_ply.elements[0].data
    # bg_face = bg_ply.elements[1].data

    with open(fg, 'rb') as f:
        fg_rows = f.readlines()

    with open(bg, 'rb') as f:
        bg_rows = f.readlines()[13:]

    # print(fg_rows[:13])

    # print(fg_rows[2])
    fg_rows[2] = ('element vertex ' + str(len(fg_vertex) + len(bg_vertex)) + '\n').encode()

    # print(fg_rows[:12])

    fg_rows = np.concatenate([fg_rows, bg_rows]).tolist()

    # print(fg_rows[:12])

    f = open(result, "wb")
    for item in fg_rows:
        f.write(item)
    f.close()

    # print(time.time() - start)

    # return

    """
    with open(result, 'w') as f:
        f.seek(0)
        f.write('ply\nformat ascii 1.0\ncomment VCGLIB generated\n')
        f.write('element vertex ')
        f.write(str(len(fg_vertex)+len(bg_vertex)))
        f.write('\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uint8 red\nproperty uint8 green\nproperty uint8 blue\n')
        f.write('property float32 nx\nproperty float32 ny\nproperty float32 nz\n')
        f.write('end_header\n')

        items = chunkIt(fg_vertex, 8)

        with Pool(len(items)) as p:
            msa = p.map(write, items)

        s = ""
        for item in msa:
            #f.write(item)
            s += item

        items = chunkIt(bg_vertex, 8) # multiprocessing.cpu_count()

        with Pool(len(items)) as p:
            msa = p.map(write, items)

        for item in msa:
            s += item


        f.write(s)
    """

if __name__ == '__main__':
    start = time.time()
    merge("scene_dense.ply", "00002_scene_dense.ply", "merged_scene_dense.ply")
    print(time.time() - start)
