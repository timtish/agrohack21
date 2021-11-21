import h5py as h5

#import os.path as fs
# path = fs.join('/data/result', 'Movie_1.h5')
# boxs = [[0.1,0.1,0.2,0.3],[0.7,0.7,0.9,0.8]]
# tracks = [[frame_idx,pig_idx,x,y],...]


def savePredict(path, boxs, masks=None, tracks=None):
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
        num_frames = len(boxs)
        # для каждого кадра
        for i in range(num_frames):
            grp = file.create_group("Frame_%d" % i)
            # количество в загоне
            num_pigs = len(boxs[i])
            grp.create_dataset('num_pigs', data=num_pigs)
            # прямоугольники
            grp.create_dataset('boxs', data=boxs[i])
            # маски
            masks_group = grp.create_group("PolyMasks")
            if masks:
                for p in range(num_pigs):
                    masks_group.create_dataset('polymask_%d' % p, data=masks[i][p])
    # перемещения и другие признаки
    # отдельным файлом
    print(tracks)
    if tracks and len(tracks) > 0:
        with h5.File(path[:-2]+'track.h5', 'w') as file:
            for pig_idx in range(max([t[1] for t in tracks])+1):
                pig_track = [[x[0], x[2], x[3]] for x in tracks if x[1] == pig_idx]
                if len(pig_track) > 0:
                    pig_group = file.create_group('pig_%d' % pig_idx)
                    pig_group.create_dataset("track", data=pig_track)
                    pig_group.create_dataset("pose", data=predict_activity(pig_track))

    #h5print(path)


def h5print(path):
    print('file:', path)
    with h5.File(path, "r") as f:
        for frame_idx in f.keys():
            frame = f[frame_idx]
            print(frame_idx, 'pigs', frame['num_pigs'][()])
            for key in frame.keys():
                if 'num_pigs' != key:
                    print('  ', key, list(frame[key]))


def predict_activity(track):
    first_frame = min([t[0] for t in track])
    last_frame = max([t[0] for t in track])
    avg_x = float(sum([t[1] for t in track]))/len(track)
    max_dev_x = max([max([t[1] for t in track])-avg_x, avg_x-min([t[1] for t in track])])
    avg_y = float(sum([t[2] for t in track]))/len(track)
    max_dev_y = max([max([t[2] for t in track])-avg_y, avg_y-min([t[2] for t in track])])
    pose = 1 if max_dev_x > 300 or max_dev_y > 300 else 0
    print(first_frame, last_frame, max_dev_x, max_dev_y, 'бегает' if pose == 1 else 'стоит/лежит')
    return [[first_frame, last_frame, pose]]

