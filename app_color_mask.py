import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel
)

# --------------------------------------------------------
# Global setup: load models/pipeline once at startup
# --------------------------------------------------------
def make_canny_condition(image: Image.Image) -> Image.Image:
    """
    Convert an input image to its Canny edges as a 3-channel PIL image.
    """
    image_array = np.array(image)
    edges = cv2.Canny(image_array, 100, 200)
    edges_3ch = np.stack([edges]*3, axis=-1)  # shape (H,W,3)
    canny_image = Image.fromarray(edges_3ch)
    return canny_image

def create_inpaint_inputs(
    lineart_image: Image.Image,
    color_mask_image: Image.Image
):
    """
    Given the lineart and a 'color mask' image, produce:
      1. an init_image that has the masked regions already filled in
      2. a binary mask (white = paint here, black = keep)
    
    We assume that any pixel in 'color_mask_image' that is neither black nor white 
    is a 'color region' which we want to fill into the line art and inpaint.
    """
    # Convert to numpy
    lineart_array = np.array(lineart_image)
    mask_array = np.array(color_mask_image)

    # Make a copy for init_image
    init_array = lineart_array.copy()

    # Binary mask will be single-channel (H,W), but eventually must convert to 3‐channel for the pipeline.
    inpaint_mask = np.zeros((init_array.shape[0], init_array.shape[1]), dtype=np.uint8)

    # Define simple helpers to check if a pixel is near black or white
    def is_black(rgb, threshold=30):
        return all(c < threshold for c in rgb)
    def is_white(rgb, threshold=225):
        return all(c > threshold for c in rgb)

    # Fill the lineart copy with the color from mask_array where needed
    height, width, _ = init_array.shape
    for y in range(height):
        for x in range(width):
            r, g, b = mask_array[y, x]
            if not is_black((r, g, b)) and not is_white((r, g, b)):
                # Mark this pixel for inpainting, and copy color from the mask
                init_array[y, x] = (r, g, b)
                inpaint_mask[y, x] = 255

    # Convert back to PIL
    init_pil = Image.fromarray(init_array)
    mask_pil = Image.fromarray(inpaint_mask).convert("RGB")
    return init_pil, mask_pil

# Load ControlNet for canny edges
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# Load the inpaint pipeline with ControlNet
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Optionally fix a seed for reproducibility
generator = torch.Generator(device="cpu").manual_seed(1)

# --------------------------------------------------------
# Inference function
# --------------------------------------------------------
def generate_colored_image(lineart_path: str, mask_path: str, prompt: str) -> Image.Image:
    """
    Given a lineart sketch, a color (not just binary) mask, and a user prompt,
    run the ControlNet inpainting pipeline so that the masked region uses
    the same color from the mask in the final image.
    """
    if not lineart_path or not mask_path:
        return None

    # Load images at 1024x1024
    lineart_image = Image.open(lineart_path).convert("RGB")
    lineart_image = lineart_image.resize((1024, 1024), Image.Resampling.LANCZOS)

    color_mask_image = Image.open(mask_path).convert("RGB")
    color_mask_image = color_mask_image.resize((1024, 1024), Image.Resampling.LANCZOS)

    # Create an init image that has the color regions spliced in,
    # plus a binary inpaint mask to say “only paint those color pixels.”
    init_image, inpaint_mask = create_inpaint_inputs(lineart_image, color_mask_image)

    # Create canny edge control image from the *original* line art
    control_image = make_canny_condition(lineart_image)

    # Run the inpaint pipeline
    result = pipe(
        prompt=prompt,
        num_inference_steps=50,
        generator=generator,
        image=init_image,       # Our image with color filled in
        mask_image=inpaint_mask, # Where to paint
        control_image=control_image,
        guidance_scale=4,
        eta=1.0
    ).images[0]

    return result

# --------------------------------------------------------
# Gradio interface definition
# --------------------------------------------------------
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Lineart Coloring App with SDXL + ControlNet")
        gr.Markdown(
            "Upload your lineart sketch and a **color** mask (where the region "
            "you want painted is filled with the actual color you want). Then "
            "provide a text prompt to steer the overall style."
        )

        with gr.Row():
            lineart_input = gr.Image(type="filepath", label="Upload Lineart Sketch")
            color_mask_input = gr.Image(type="filepath", label="Upload Color Mask")

        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="E.g. 'A detailed illustration in an anime style, glossy finish.'"
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
