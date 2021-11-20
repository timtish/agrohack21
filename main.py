"""
При запуске обрабатывает все файлы из папки source_files
Результат помещает в папку out_files, добавляя к имени "out_"
Если файл с таким названием уже обрабатывался, то его не трогает.
Видео во время обработки отображается.
"""
# Модуль с функицями
import pigs
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Допустимые форматы
img_type_list = ['.jpg', '.jpeg', '.png']
vid_type_list = ['.mp4', '.avi', '.mkv']


def process(source_PATH, out_PATH, model_PATH):
    """
    @param: source_PATH путь к каталогу с файлами
    @param: out_PATH путь результатам
    @param: model_PATH Путь к модели
    """
    # Создадим папки для файлов, если их нет
    if not (source_PATH in os.listdir('.')):
        os.mkdir(source_PATH)
    if not (out_PATH in os.listdir('.')):
        os.mkdir(out_PATH)

    # В папке должен быть файл модели
    assert model_PATH in os.listdir('.'), 'В папке программы должен быть файл модели'

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_PATH))
    out_files = sorted(os.listdir(out_PATH))
    # Раздельные списки для картинок и видео
    img_files = []
    vid_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # print(f,filename,file_extension)
        if not (('out_'+f) in out_files):
            if file_extension in img_type_list:
                img_files.append(f)
            if file_extension in vid_type_list:
                vid_files.append(f)

    # Обрабатываем картинки
    for img in img_files:
        # полные пути к файлам
        img_FILE = source_PATH + '/' + img
        out_FILE = out_PATH + '/' + 'out_' + img
        # Вызов функции предикта
        _ = pigs.detect(model_PATH, img_FILE, out_FILE)

    # Обрабатываем видео
    for vid in vid_files:
        # полные пути к файлам
        vid_FILE = source_PATH + '/' + vid
        out_FILE = out_PATH + '/' + 'out_' + vid
        # Вызов функции предикта
        _ = pigs.detect(model_PATH, vid_FILE, out_FILE)

    # Сообщаем что обработали
    if len(img_files) == 0:
        print('Нет картинок для обработки.')
    else:
        print('Обработали {0} картинок.'.format(len(img_files)))
    if len(vid_files) == 0:
        print('Нет видео для обработки.')
    else:
        print('Обработали {0} видео.'.format(len(vid_files)))


if __name__ == '__main__':
    source_PATH = 'source_files' if len(sys.argv) <= 1 else sys.argv[1]
    out_PATH = 'out_files' if len(sys.argv) <= 2 else sys.argv[2]
    model_PATH = 'best.pt' if len(sys.argv) <= 3 else sys.argv[3]
    process(source_PATH, out_PATH, model_PATH)
