import torch
import cv2
from yolov5 import YOLOv5

import torch
print(torch.cuda.is_available())


# Modeli yükle (pre-trained COCO dataset ile)
model_path = 'yolov5s.pt'  # Küçük, hızlı model. Alternatifler: yolov5m.pt, yolov5l.pt, yolov5x.pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLOv5(model_path, device)

# Görsel oku
img = cv2.imread('test.jpg')  # test.jpg yerine kendi görselini koy

# BGR'dan RGB'ye çevir (model için gerekebilir)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Nesne tespiti yap
results = model.predict(img_rgb)

# Sonuçları yazdır
print(results)

# Sonuçları çiz ve göster
results.show()  # Alternatif: results.save(save_dir='outputs/')

# Eğer sonuçları OpenCV ile göstermek istersen:
result_img = results.render()[0]
cv2.imshow('Detection Result', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
