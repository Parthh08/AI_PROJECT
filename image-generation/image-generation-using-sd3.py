import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

def image_generation(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline.to(device)
    
    enhanced_prompt = f"high quality, detailed {prompt}"
    
    image = pipeline(prompt=enhanced_prompt,
                     negative_prompt="blurred, ugly, watermark, low resolution, blurry, distorted proportions, bad anatomy",
                     num_inference_steps=25,
                     height=512,  
                     width=512, 
                     guidance_scale=8.5,  
                     ).images[0]
    return image

#image_generation("A magician cat doing spell")


interface = gr.Interface(
    fn = image_generation,
    inputs = gr.Textbox(lines=2,placeholder="Enter your prompt"),
    outputs = gr.Image(type="pil"),
    title = "GEN AI"


)
interface.launch()