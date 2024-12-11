import torch
import clip
from PIL import Image
Lable=["D1382890", "D1311190", "D1322290", "D1333890",'dog']
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("image.png")).unsqueeze(0).to(device)
text = clip.tokenize(Lable).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:")
for i, prob in enumerate(probs[0]):
    print(f"{Lable[i]}: {prob:.4f}")