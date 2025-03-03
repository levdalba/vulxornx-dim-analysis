from PIL import Image
import os

input_folder = "dataset/tif"  # Change this to your actual image folder
output_folder = "dataset/jpg"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".tif"):
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path)
        jpg_path = os.path.join(output_folder, file.replace(".tif", ".jpg"))
        img.convert("RGB").save(jpg_path, "JPEG")
        print(f"Converted {file} to {jpg_path}")

print("Conversion complete!")
