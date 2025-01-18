# Lineart Coloring App

This project is a Gradio-based web application that allows users to generate colored images from lineart sketches and color masks using a diffusion model and controlnet, you can specify the type of color in you want the region of interset to have within the prompt.

## Features
- Upload lineart and binary mask images.
- Add a prompt to specify color, etc.
- Generate a colored image using a pretrained diffusion model.

## Installation

### Prerequisites


1. Set up a virtual environment:
   ```
   python -m venv env
   
   # on ubuntu
   source env/bin/activate
   
   # On Windows use 
   env\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. You can change the model,device and inference steps in the `constants.py`, the current configuration is as follows:
   ```python
   GENERATION_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
   CONTROL_NET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
   DEVICE = "cuda"
   NUM_INFERENCE_STEPS = 20
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open the provided local URL in your web browser to use the app.

## Input and Output

- **Input:**
  - `lineart_path`: File path to the lineart image.
  - `color_mask_path`: File path to the color mask image (The area you are interested in coloring).
  - `Prompt`: How you would like to color to be and other details

- **Output:**
  - Generated colored image displayed on the Gradio interface.


## Sample input and generated output

### Input line art image
![Line art image](/Images/Base_Image.jpeg)

### Input mask image
![Binary mask image](/Images/Mask_image.png)

### Input prompt 
```
a girl with green retina
```
### Generated output
![Generated Output](/Images/green_eye.png)
