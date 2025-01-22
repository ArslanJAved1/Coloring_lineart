from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2

# --------------------------------------------------------
# Global setup: load models/pipelines once at startup
# --------------------------------------------------------

# Function to create Canny edges (edge control)
def make_canny_condition(image: Image.Image) -> Image.Image:
    """
    Convert an input image to its Canny edges as a 3-channel PIL image.
    """
    image_array = np.array(image)
    edges = cv2.Canny(image_array, 100, 200)
    edges_3ch = np.stack([edges] * 3, axis=-1)  # shape (H,W,3)
    canny_image = Image.fromarray(edges_3ch)
    return canny_image

# Load ControlNet for edges
controlnet_edge = ControlNetModel.from_pretrained(
    "xinsir/controlnet-scribble-sdxl-1.0",
    torch_dtype=torch.float16
)

# Load ControlNet for segmentation (to enforce color masking)
controlnet_segmentation = ControlNetModel.from_pretrained(
    "SargeZT/sdxl-controlnet-seg",
    torch_dtype=torch.float16
)

# Load the main Stable Diffusion XL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=[controlnet_edge, controlnet_segmentation],  # Multi-ControlNet
    torch_dtype=torch.float16
)
pipe.to("cuda")

# --------------------------------------------------------
# Updated Inference Function for Multi-ControlNet
# --------------------------------------------------------

def generate_colored_image(lineart_path: str, mask_path: str, prompt: str) -> Image.Image:
    """
    Given a lineart sketch, a color (not just binary) mask, and a user prompt,
    run the Multi-ControlNet pipeline so that:
      1. Lineart defines the structure.
      2. Color mask controls the colors in regions using segmentation.
    """
    if not lineart_path or not mask_path:
        return None

    # Load and preprocess images
    lineart_image = Image.open(lineart_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    color_mask_image = Image.open(mask_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)

    # Create control image for edges (Canny)
    control_image_edge = make_canny_condition(lineart_image)

    # Use the color mask directly for segmentation control
    control_image_segmentation = color_mask_image

    # Run the pipeline with two control images
    result = pipe(
        prompt=prompt,
        num_inference_steps=50,
        generator=torch.Generator("cuda").manual_seed(42),  # For reproducibility
        image=[lineart_image,lineart_image],  # The main image
        mask_image=color_mask_image,  # Mask input
        control_image=[control_image_edge, control_image_segmentation],  # Multi-ControlNet
        guidance_scale=7.5,
        controlnet_conditioning_scale=[1.5, 1.0],  # Scales for edge and segmentation ControlNet
    ).images[0]

    return result

# --------------------------------------------------------
# Gradio Interface Definition
# --------------------------------------------------------

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-ControlNet Lineart Coloring App")
        gr.Markdown(
            "Upload your lineart sketch and a **color mask** (regions filled "
            "with the desired color), then provide a text prompt for style. "
            "This app uses a multi-ControlNet setup for both edge and color control."
        )

        with gr.Row():
            lineart_input = gr.Image(type="filepath", label="Upload Lineart Sketch")
            color_mask_input = gr.Image(type="filepath", label="Upload Color Mask")

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="E.g. 'A fantasy anime-style illustration with a glossy finish.'"
        )

        output_image = gr.Image(label="Generated Colored Image")

        generate_button = gr.Button("Generate Image")
        generate_button.click(
            fn=generate_colored_image,
            inputs=[lineart_input, color_mask_input, prompt_input],
            outputs=output_image
        )

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(share=True,debug=True)
