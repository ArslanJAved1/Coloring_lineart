import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from constants import Constants
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel

def load_model() -> StableDiffusionXLControlNetInpaintPipeline:
    """
    Load the pretrained diffusion model specified in the constants module.
    
    Returns:
        DiffusionPipeline: The loaded diffusion pipeline moved to the specified device.
    """
    # Load ControlNet for canny
    controlnet = ControlNetModel.from_pretrained(
        Constants.CONTROL_NET_MODEL_ID,
        torch_dtype=torch.float16
    )

    # Load the inpaint pipeline with ControlNet
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        Constants.GENERATION_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe.to(Constants.DEVICE)
    return pipe

pipe = load_model()

# Optionally fix a seed for reproducibility
generator = torch.Generator(device=Constants.DEVICE).manual_seed(1)

def make_canny_condition(image: Image.Image) -> Image.Image:
    """
    Convert an input image to its Canny edges as a 3-channel PIL image.
    Args:
        image: Image.Image: Input image
    Returns:
        canny_image : returns the image edges.
    """
    image_array = np.array(image)
    edges = cv2.Canny(image_array, 100, 200)
    edges_3ch = np.stack([edges]*3, axis=-1)
    canny_image = Image.fromarray(edges_3ch)
    return canny_image


def generate_colored_image(lineart_path: str, mask_path: str, prompt: str) -> Image.Image:
    """
    Given a lineart sketch, a binary mask, and a user prompt, run the ControlNet 
    inpainting pipeline to produce a colored image.
    
    Args:
        lineart_path (str): File path to the uploaded lineart image.
        mask_path (str): File path to the uploaded binary mask.
        prompt (str): The text prompt describing desired color/style.
    
    Returns:
        PIL.Image.Image: The generated colored image.
    """
    # Basic validation: check if files exist
    if not lineart_path or not mask_path:
        return None
    
    # Load images
    lineart_image = Image.open(lineart_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")

    # Resize both to 1024x1024 to match your original setup
    lineart_image = lineart_image.resize((1024, 1024), Image.Resampling.LANCZOS)
    mask_image = mask_image.resize((1024, 1024), Image.Resampling.LANCZOS)

    # Create canny edge control image
    control_image = make_canny_condition(lineart_image)

    # Run the inpaint pipeline
    result = pipe(
        prompt=prompt,
        num_inference_steps=Constants.NUM_INFERENCE_STEPS,
        generator=generator,
        eta=1.0,
        image=lineart_image,
        mask_image=mask_image,
        control_image=control_image
    ).images[0]

    return result


def gradio_interface() ->gr.Blocks :
    """
    Create and launch a Gradio interface for the Lineart Coloring App.
    
    Returns:
        gr.Blocks: The Gradio Blocks interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Lineart Coloring App with StableDiffusion-XL")
        gr.Markdown("Upload your lineart sketch and a binary mask, then provide a prompt describing the colors or style you want.")

        with gr.Row():
            lineart_input = gr.Image(type="filepath", label="Upload Lineart Sketch")
            color_mask_input = gr.Image(type="filepath", label="Upload Binary Mask")

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="E.g. 'A vibrant, colorful illustration' or 'A pastel watercolor style.'"
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
    demo.launch(share=True)