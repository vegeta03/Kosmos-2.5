import spaces
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw
import re
import gradio as gr

repo = "microsoft/kosmos-2.5"
device = "cuda"

config = AutoConfig.from_pretrained(repo)
dtype = torch.float16

model = AutoModelForVision2Seq.from_pretrained(
    repo, device_map=device, torch_dtype=dtype, config=config
)

processor = AutoProcessor.from_pretrained(repo)

prompt = "<ocr>"  # Options are '<ocr>' and '<md>'


@spaces.GPU
def process_image(image_path):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    height, width = inputs.pop("height"), inputs.pop("width")
    raw_width, raw_height = image.size
    scale_height = raw_height / height
    scale_width = raw_width / width

    inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
    inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return postprocess(generated_text, scale_height, scale_width, image)


def postprocess(y, scale_height, scale_width, original_image):
    y = y.replace(prompt, "")

    if "<md>" in prompt:
        return y, original_image

    pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    bboxs_raw = re.findall(pattern, y)

    lines = re.split(pattern, y)[1:]
    bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    bboxs = [[int(j) for j in i] for i in bboxs]

    info = ""

    # Create a copy of the original image to draw on
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

            # Draw rectangle on the image
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    return image_with_boxes, info


iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Image(type="pil", label="Image with Bounding Boxes"),
        gr.Textbox(label="Extracted Text"),
    ],
)

iface.launch()
