import h5py as h5

#import os.path as fs
# path = fs.join('/data/result', 'Movie_1.h5')
# boxs = [[0.1,0.1,0.2,0.3],[0.7,0.7,0.9,0.8]]


def savePredict(path, boxs, masks=None):
    """
    Для минутного ролика имеем один файл. Внутри файла группы с
    Frame_i, где i -- это номер фрейма по порядку.
    num_pigs -- число свиней на кадре. list <Int [1]>
    boxs -- список из боксов на каждом фрейме; list <array [num_pigs, 4] >, [x_min, y_min, x_max, y_max]
    masks -- список из списка вырезанных масок (избавляемся от лишних нулей,
        оставляем только то, что внутри соответствуюещй коробки) list < list [num_pigs, x_shape, y_shape] >
    Для трекинга, boxs и masks сразу сохраняем в соответвтующем порядке.
    """
    with h5.File(path, 'w') as file:
        num_frames = len(boxs)  # количество фреймов
        for i in range(num_frames):
            grp = file.create_group("Frame_%d" % i)
            num_pigs = len(boxs[i])
            grp.create_dataset('num_pigs', data=num_pigs)
            grp.create_dataset('boxs', data=boxs[i])
            masks_group = grp.create_group("PolyMasks")
            if masks:
                for p in range(num_pigs):
                    masks_group.create_dataset('polymask_%d' % p, data=masks[i][p])
    h5print(path)


def h5print(path):
    print('file:', path)
    with h5.File(path, "r") as f:
        for frame_idx in f.keys():
            frame = f[frame_idx]
            print(frame_idx, 'pigs', frame['num_pigs'][()])
            for key in frame.keys():
                if 'num_pigs' != key:
                    print('  ', key, list(frame[key]))

