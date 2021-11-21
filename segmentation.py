import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.util import img_as_ubyte
from utils.data_loading import BasicDataset
from PIL import Image
import matplotlib.pyplot as plt

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)


def predict_img(full_img,
                net=net,
                device='cpu',
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


# local test
if __name__ == '__main__':
    # открываем видео
    cap = cv2.VideoCapture('/data/dev/ML/hackatons/ferma/train/Movie_1.mkv')
    if not cap.isOpened():
        print("Ошибка открытия файла видео")

    # Рассчитаем коэффициент для изменения размера
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10

    # Получаем фреймы пока видео не закончится
    frame_out_idx = 0
    while cap.isOpened():
        cap.read()
        cap.read()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame)
        mask = predict_img(pil_im)  # (2, 1700, 1700)
        print(mask)

        break
