from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("../../productinsight_ai/src/data/images", exist_ok=True)

colors = [
    ('1', (0,0,0), 'Black'),
    ('2', (255,255,255), 'White'),
    ('3', (255,0,0), 'Red'),
    ('4', (0,0,255), 'Blue'),
    ('5', (0,128,0), 'Green'),
]

for img_id, color, name in colors:
    img = Image.new("RGB", (300,400), color)
    draw = ImageDraw.Draw(img)

    text_color = (255,255,255) if sum(color)<300 else (0,0,0)
    draw.text((50,180), f"{name}\nProduct {img_id}", fill=text_color)
    img.save(f"data/images/image_{img_id}.jpg")
    print(f"image_{img_id}.jpg created")