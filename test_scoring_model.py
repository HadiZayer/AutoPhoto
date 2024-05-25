from scoring_model import ScoringModel
from PIL import Image

test_img_path = 'cat.jpg'
img_pil = Image.open(test_img_path).convert('RGB')
scoring_model = ScoringModel(gpu=False)  # if gpu is available, switch to Truethe image on cuda
transform = scoring_model.transform

img_t = transform(img_pil)
score = scoring_model.forward([img_t])  # takes list of tensor images, output tensor of scores
print(score)

