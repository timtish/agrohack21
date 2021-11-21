"""
Модуль с функцией предикта по картинке
"""
def detect(weight_path, source_img_path, save_img_path='out.jpg'):
    """
    Args:
        weight_path: путь до файла весов модели
        source_img_path: путь до файла, который классифицируем
        save_img_path: путь до файла, куда сохраняем результат.
    """
    import time
    from pathlib import Path

    import cv2
    import torch
    import torch.backends.cudnn as cudnn
    from numpy import random

    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
        set_logging
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized

    from utils.h5export import savePredict
    from tracks import get_tracks_by_boxes

    # размер обрабатываемой картинки
    size_img = 1024

    # Этот словарь - это все параметры скрипта detect.py
    opt = dict(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', img_size=size_img,
               iou_thres=0.45, project=save_img_path, save_conf=False, save_txt=False, save_h5=True,
               source=source_img_path, view_img=False, weights=[weight_path])
    source, weights, view_img, save_txt, imgsz, save_h5 = opt['source'], opt['weights'], opt['view_img'], \
                                                          opt['save_txt'], opt['img_size'], opt['save_h5']
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    # Directories
    save_dir = Path(opt['project'])

    # Initialize
    set_logging()
    device = select_device(opt['device'])
    #device = select_device('cpu')  # ПРИНУДИТЕЛЬНО БЕРЕМ CPU
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = False
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # НАДПИСИ на РУССКОМ
    names = ['Свинка']
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # ЧТОБЫ БЫЛ ОДИНАКОВЫЙ ЦВЕТ РАМОК
    #colors = [[0,255,0],[255,0,0]]
    colors = [[200,150,150],[200,150,150]]
    # (так же см.правки в plots.py)


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    predicted_boxes_all_frames = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]
        #pred = model(img, augment=opt['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'],
                                   agnostic=opt['agnostic_nms'])
        t2 = time_synchronized()

        # Process detections
        predicted_boxes = []
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir)
            txt_path = str(save_dir.parent / 'labels' / save_dir.stem) + (
                '_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                predicted_boxes.append([b.tolist() for b in det[:, :4].cpu().numpy().astype('int')])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'vp80'  # output video codec 'mp4v'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        predicted_boxes_all_frames.append(predicted_boxes)

    masks = []
    if save_h5:
        tracks = get_tracks_by_boxes(predicted_boxes_all_frames)
        savePredict(str(save_dir) + '.h5', predicted_boxes_all_frames, masks, tracks)

    if save_txt or save_img:
        print('Results saved to %s' % save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))

    return save_img_path


