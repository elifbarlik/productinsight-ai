import os

from PIL import Image, ImageDraw

os.makedirs("src/data/images", exist_ok=True)

colors = [
    ('1', (0,0,0), 'Black'),
    ('2', (255,255,255), 'White'),
    ('3', (255,0,0), 'Red'),
    ('4', (0,0,255), 'Blue'),
    ('5', (0,128,0), 'Green'),
    ('6', (0,0,128), 'Navy'),
    ('7', (139,69,19), 'Brown'),
    ('8', (128,128,128), 'Gray'),
    ('9', (255,182,193), 'Pink'),
    ('10', (30,30,30), 'Charcoal'),
    ('11', (210,180,140), 'Beige'),
    ('12', (40,40,40), 'Dark'),
    ('13', (100,149,237), 'Cornflower'),
    ('14', (210,180,140), 'Tan'),
    ('15', (105,105,105), 'DimGray'),
]

for img_id, color, name in colors:
    img = Image.new("RGB", (300,400), color)
    draw = ImageDraw.Draw(img)

    text_color = (255,255,255) if sum(color)<300 else (0,0,0)
    draw.text((50,180), f"{name}\nProduct {img_id}", fill=text_color)
    img.save(f"src/data/images/{img_id}.jpg")
    print(f"{img_id}.jpg created")

print(f"\nDone! {len(colors)} images created.")