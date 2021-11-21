import sys
import cv2
import h5py as h5
import numpy as np


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
    while cap.isOpened():

        predicted_boxes = []
        with h5.File(h5file_path, "r") as f:
            for frame_idx in f.keys():
                frame = f[frame_idx]
                predicted_boxes = [bbs.tolist() for bbs in frame['boxs']]
                predicted_boxes = predicted_boxes[0] if len(predicted_boxes) else []

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
                for t in track:
                    cv2.circle(frame, (t[1], t[2]), 3, (0, 0, 1))
                pts = np.array(track, np.int32)
                pts.sort(0)
                pts = pts[:, [2, 1]].reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (40, 160, 40), 3)

        # visualize detections
        for b in predicted_boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (51, 204, 51), 1)

        #for b in results.xyxy:
        #    score = b[5]
        #    cv2.putText(frame, "{:.2f}".format(score), (b[1], b[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        pigs_number = "pigs: {:d}".format(len(predicted_boxes))
        cv2.putText(frame, pigs_number, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

        # Запись кадра видео
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    annotate_video_by_h5(sys.argv[1], sys.argv[2], sys.argv[3])
