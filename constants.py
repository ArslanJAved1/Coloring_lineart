from dataclasses import dataclass
@dataclass(frozen=True)
class Constants:
    GENERATION_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROL_NET_MODEL_ID: str = "diffusers/controlnet-canny-sdxl-1.0"
    DEVICE: str = "cuda"
    NUM_INFERENCE_STEPS: int = 20
