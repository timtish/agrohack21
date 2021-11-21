import h5py as h5
import numpy as np

from utils.trackingutils import Sort
# https://github.com/abewley/sort/blob/master/sort.py


def get_tracks_by_boxes(boxes_all_frames):
    tracks = []
    total_frames = len(boxes_all_frames)
    tracker = Sort(cfg={'max_age': total_frames})
    for frame_idx in range(1, total_frames):
        dets = boxes_all_frames[frame_idx]
        trackers = tracker.update(np.array(dets))
        for d in trackers:
            tracks.append([frame_idx, int(d[4]), int((d[0]+d[2])/2), int((d[1]+d[3])/2)])
    return tracks


def get_tracks(h5boxes_path):
    boxes_all_frames = []
    with h5.File(h5boxes_path, "r") as h5file:
        for frame_idx in h5file.keys():
            frame = h5file[frame_idx]
            dets = [bbs.tolist() for bbs in frame['boxs']]
            if len(dets):
                dets = [d+[0.98] for d in dets[0]]
            boxes_all_frames.append(dets)

    total_frames = len(boxes_all_frames)
    print(total_frames, h5boxes_path)

    tracker = Sort(cfg={'max_age': total_frames})

    for frame_idx in range(1, total_frames):
        dets = boxes_all_frames[frame_idx]
        #print(frame_idx, dets)
        trackers = tracker.update(np.array(dets))
        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame_idx,
                        d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]))
            # add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1]


if __name__ == '__main__':
    get_tracks('/data/dev/ML/hackatons/ferma/train6/out_files/out_Movie_6.mkv.h5')
