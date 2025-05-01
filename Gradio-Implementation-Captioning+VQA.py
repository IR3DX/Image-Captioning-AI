# Install libraries
!pip install transformers gradio Pillow torch torchvision torchaudio -q

# Import libraries
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import gradio as gr

# Load models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Image Captioning
def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Visual Question Answering
def answer_question(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt")
    outputs = vqa_model.generate(**inputs)
    answer = vqa_processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# Captioning interface
caption_tab = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Text(label="Caption"),
    title="Image Captioning",
    description="Upload an image to generate a descriptive caption"
)

# VQA interface
vqa_tab = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Text(label="Ask a Question")
    ],
    outputs=gr.Text(label="Answer"),
    title="Visual Question Answering",
    description="Upload an image and ask a question about it using BLIP."
)

# Combine in tabs
gr.TabbedInterface([caption_tab, vqa_tab], tab_names=["Image Captioning", "VQA"]).launch(share=True)
