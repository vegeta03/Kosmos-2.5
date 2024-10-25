import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw
import re
import gradio as gr
from pathlib import Path
import shutil

# Constants
MODEL_REPO = "microsoft/kosmos-2.5"
MODEL_CACHE_DIR = Path("./model_cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def setup_model_directory():
    """Create and setup model cache directory if it doesn't exist."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE_DIR

def load_or_download_model():
    """Load model from cache or download if not available."""
    try:
        # Setup cache directory
        cache_dir = setup_model_directory()
        
        print(f"Loading model from {MODEL_REPO}...")
        
        # Load configuration
        config = AutoConfig.from_pretrained(
            MODEL_REPO,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Load model with caching
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_REPO,
            device_map=DEVICE,
            torch_dtype=DTYPE,
            config=config,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Load processor with caching
        processor = AutoProcessor.from_pretrained(
            MODEL_REPO,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        print("Model loaded successfully!")
        return model, processor
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def cleanup_cache():
    """Cleanup temporary cache files."""
    try:
        cache_files = MODEL_CACHE_DIR.glob("**/tmp*")
        for file in cache_files:
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
    except Exception as e:
        print(f"Warning: Cache cleanup failed: {str(e)}")

# Initialize model and processor
model, processor = load_or_download_model()

def process_image(image_path, task, num_beams, max_new_tokens, temperature):
    prompt = "<ocr>" if task == "OCR" else "<md>"
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    height, width = inputs.pop("height"), inputs.pop("width")
    raw_width, raw_height = image.size
    scale_height = raw_height / height
    scale_width = raw_width / width

    inputs = {k: v.to(DEVICE) if v is not None else None for k, v in inputs.items()}
    inputs["flattened_patches"] = inputs["flattened_patches"].to(DTYPE)

    generated_ids = model.generate(
        **inputs,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return postprocess(generated_text, scale_height, scale_width, image, prompt)


def postprocess(y, scale_height, scale_width, original_image, prompt):
    y = y.replace(prompt, "")

    if "<md>" in prompt:
        return original_image, y

    pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    bboxs_raw = re.findall(pattern, y)

    lines = re.split(pattern, y)[1:]
    bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    bboxs = [[int(j) for j in i] for i in bboxs]

    info = ""

    image_with_boxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for i in range(len(lines)):
        box = bboxs[i]
        x0, y0, x1, y1 = box

        if not (x0 >= x1 or y0 >= y1):
            x0 = int(x0 * scale_width)
            y0 = int(y0 * scale_height)
            x1 = int(x1 * scale_width)
            y1 = int(y1 * scale_height)
            info += f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{lines[i]}\n"

            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    return image_with_boxes, info


iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),
        gr.Radio(["OCR", "Markdown"], label="Task", value="OCR"),
        gr.Slider(1, 10, value=4, step=1, label="Number of Beams"),
        gr.Slider(100, 4000, value=2048, step=100, label="Max New Tokens"),
        gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Temperature"),
    ],
    outputs=[
        gr.Image(type="pil", label="Image with Bounding Boxes (OCR only)"),
        gr.Textbox(label="Extracted Text / Markdown"),
    ],
    title="Kosmos 2.5 OCR and Markdown Generator",
    description="""Generate OCR results or Markdown from images using Kosmos 2.5.
    Uses the Kosmos 2.5 [PR Branch](https://github.com/huggingface/transformers/pull/31711) of the Transformers library for inference.
    I don't know if the parameters do much of anything, but they're available for tweaking just in case.""",
)

iface.launch()
