import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Image URL to raw format
img_url = 'https://t3.ftcdn.net/jpg/02/05/72/26/360_F_205722613_jUUqRds7MJOjGGYFjPBzhF2883z0kyiJ.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# The question you want to ask about the image
question = "How many dogs are in the image" #You can also try: "Are there any rabbits in the image?"

# prepare inputs 
inputs = processor(raw_image, question, return_tensors="pt")

# Generate the answer from the model
out = model.generate(**inputs)

# Decode and print the answer to the question
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")
