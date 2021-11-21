import sys
import cv2
import h5py as h5
import numpy as np
from utils.plots import plot_one_box

TRACK_TIME_SEC = 5


def annotate_video_by_h5(src_file_path, h5file_path, dst_video_path):
    """
    Функция наложения разметки на видео

    @param: src_file_path исходное видео 1700x1700
    @param: h5file_path информация о боксах и перемещениях
    @param: dst_video_path файл куда сохранить видео

    """
    print('annotate video by h5 ', src_file_path, h5file_path, '->', dst_video_path)
    # открываем видео
    cap = cv2.VideoCapture(src_file_path)
    if not cap.isOpened():
        print("Ошибка открытия файла видео")

    # Рассчитаем коэффициент для изменения размера
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(dst_video_path, fourcc, fps, (width, height))

    # Получаем фреймы пока видео не закончится
    frame_out_idx = 0
    while cap.isOpened():

        predicted_boxes = []
        with h5.File(h5file_path, "r") as f:
            for frame_idx in f.keys():
                frame = f[frame_idx]
                bb = [bbs.tolist() for bbs in frame['boxs']]
                predicted_boxes.append(bb[0] if len(bb) else [])

        pigs_tracks = []
        with h5.File(h5file_path[:-2]+'track.h5', "r") as f:
            for frame_idx in f.keys():
                frame = f[frame_idx]
                pigs_tracks.append([bbs.tolist() for bbs in frame['track']])

        ret, frame = cap.read()
        if not ret:
            break

        # visualize tracks
        for track in pigs_tracks:
            if len(track) > 0:
                #for t in track:
                #    cv2.circle(frame, (t[1], t[2]), 3, (0, 0, 1))
                pts = np.array(track, np.int32)
                pts.sort(0)
                pts = pts[max(0, frame_out_idx-fps*TRACK_TIME_SEC):frame_out_idx, [1, 2]].reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (40, 160, 40), 3)

        # visualize detections
        for b in predicted_boxes[frame_out_idx]:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (51, 204, 51), 1)
            plot_one_box((b[0], b[1], b[2], b[3]), frame, color=[200, 150, 150], line_thickness=3)

        #for b in results.xyxy:
        #    score = b[5]
        #    cv2.putText(frame, "{:.2f}".format(score), (b[1], b[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        pigs_number = "pigs: {:d}".format(len(predicted_boxes))
        cv2.putText(frame, pigs_number, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        # Запись кадра видео
        out.write(frame)
        frame_out_idx += 1

    cap.release()
    out.release()


def annotate_video_by_arrays(src_file_path, dst_video_path, predicted_boxes_all_frames, masks, tracks):
    """
    Функция наложения разметки на видео

    @param: src_file_path исходное видео 1700x1700
    @param: h5file_path информация о боксах и перемещениях
    @param: dst_video_path файл куда сохранить видео

    """
    print('annotate video')
    # открываем видео
    cap = cv2.VideoCapture(src_file_path)
    if not cap.isOpened():
        print("Ошибка открытия файла видео")

    # Рассчитаем коэффициент для изменения размера
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(dst_video_path, fourcc, fps, (width, height))

    # Получаем фреймы пока видео не закончится
    frame_out_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # visualize tracks
        for pig_idx in range(max([t[1] for t in tracks])+1):
            track = [[x[0], x[2], x[3]] for x in tracks
                     if x[1] == pig_idx and frame_out_idx - fps * TRACK_TIME_SEC < x[0] <= frame_out_idx]
            if len(track) > 0:
                t = track[-1]
                cv2.circle(frame, (t[1], t[2]), 9, (0, 0, 0))
                pts = np.array(track, np.int32)
                #pts.sort(0)
                pts = pts[:, [1, 2]].reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (40, 160, 40), 3)

        # visualize detections
        if len(predicted_boxes_all_frames) > frame_out_idx:
            predicted_boxes = predicted_boxes_all_frames[frame_out_idx]
            for b in predicted_boxes:
                #cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (51, 204, 51), 1)
                plot_one_box((b[0], b[1], b[2], b[3]), frame, color=[200, 150, 150], line_thickness=3)
        # else:
            # print('out of index predicted_boxes_all_frames', frame_out_idx)

        #for b in results.xyxy:
        #    score = b[5]
        #    cv2.putText(frame, "{:.2f}".format(score), (b[1], b[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        pigs_number = "count: {:d}".format(len(predicted_boxes))
        cv2.putText(frame, pigs_number, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 150, 150), 2)

        # Запись кадра видео
        out.write(frame)
        frame_out_idx += 1

    cap.release()
    out.release()


if __name__ == '__main__':
    annotate_video_by_h5(sys.argv[1], sys.argv[2], sys.argv[3])
