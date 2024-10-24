import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageDraw, ImageSequence
import re
import gradio as gr
import os
import tempfile
from functional import seq
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class ProcessingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    repo: str = "microsoft/kosmos-2.5"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@dataclass
class PageResult:
    page_num: int
    image: Optional[Image.Image]
    text: str
    error: Optional[str] = None

class KosmosProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = self._initialize_model()
        self.processor = AutoProcessor.from_pretrained(config.repo)

    def _initialize_model(self):
        model_config = AutoConfig.from_pretrained(self.config.repo)
        return AutoModelForVision2Seq.from_pretrained(
            self.config.repo,
            device_map=self.config.device,
            torch_dtype=self.config.dtype,
            config=model_config
        )

    def process_page(self, page_data: Tuple[int, Image.Image], task: str, 
                    num_beams: int, max_new_tokens: int, temperature: float) -> PageResult:
        page_num, page = page_data
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=True) as tmp_file:
                page.save(tmp_file.name)
                image_with_boxes, text_output = self._process_single_image(
                    tmp_file.name, task, num_beams, max_new_tokens, temperature
                )
                return PageResult(page_num + 1, image_with_boxes, text_output)
        except Exception as e:
            return PageResult(page_num + 1, None, "", str(e))

    def _process_single_image(self, image_path: str, task: str, 
                            num_beams: int, max_new_tokens: int, 
                            temperature: float) -> Tuple[Image.Image, str]:
        prompt = "<ocr>" if task == "OCR" else "<md>"
        image = Image.open(image_path)
        inputs = self._prepare_inputs(image, prompt)
        
        generated_text = self._generate_text(inputs, num_beams, max_new_tokens, temperature)
        return self._postprocess_output(generated_text, inputs, image, prompt)

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs['scale_factors'] = {
            'height': image.size[1] / inputs.pop("height"),
            'width': image.size[0] / inputs.pop("width")
        }
        inputs = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        if self.config.device == "cuda":
            inputs["flattened_patches"] = inputs["flattened_patches"].to(self.config.dtype)
        return inputs

    def _generate_text(self, inputs: Dict[str, Any], num_beams: int, 
                      max_new_tokens: int, temperature: float) -> str:
        context_manager = (torch.cuda.amp.autocast() if self.config.device == "cuda" 
                         else torch.no_grad())
        with context_manager:
            generated_ids = self.model.generate(
                **{k: v for k, v in inputs.items() if k != 'scale_factors'},
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _postprocess_output(self, text: str, inputs: Dict[str, Any], 
                          image: Image.Image, prompt: str) -> Tuple[Image.Image, str]:
        # Existing postprocess logic converted to functional style
        return (seq(text)
                .map(lambda t: t.replace(prompt, ""))
                .map(lambda t: self._process_bboxes(t, inputs['scale_factors'], image, prompt))
                .head())

    def _process_bboxes(self, text: str, scale_factors: Dict[str, float], 
                       image: Image.Image, prompt: str) -> Tuple[Image.Image, str]:
        if "<md>" in prompt:
            return image, text

        pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
        bboxs_raw = re.findall(pattern, text)
        lines = re.split(pattern, text)[1:]
        
        return (seq(zip(bboxs_raw, lines))
                .map(lambda x: self._process_single_bbox(x, scale_factors, image))
                .reduce(self._combine_bbox_results))

    def _process_single_bbox(self, bbox_line: Tuple[str, str], 
                           scale_factors: Dict[str, float], 
                           image: Image.Image) -> Tuple[Image.Image, str]:
        bbox_raw, line = bbox_line
        coords = [int(x) for x in re.findall(r"\d+", bbox_raw)]
        
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            return image, ""

        scaled_coords = [
            int(coords[0] * scale_factors['width']),
            int(coords[1] * scale_factors['height']),
            int(coords[2] * scale_factors['width']),
            int(coords[3] * scale_factors['height'])
        ]
        
        img_copy = image.copy()
        ImageDraw.Draw(img_copy).rectangle(scaled_coords, outline="red", width=2)
        
        return img_copy, f"{scaled_coords[0]},{scaled_coords[1]},{scaled_coords[2]}," \
               f"{scaled_coords[1]},{scaled_coords[2]},{scaled_coords[3]}," \
               f"{scaled_coords[0]},{scaled_coords[3]},{line}\n"

    def _combine_bbox_results(self, acc: Tuple[Image.Image, str], 
                            curr: Tuple[Image.Image, str]) -> Tuple[Image.Image, str]:
        return curr[0], acc[1] + curr[1]

def process_multipage_tiff(image_path: str, task: str, num_beams: int, 
                          max_new_tokens: int, temperature: float) -> Tuple[Image.Image, str]:
    config = ProcessingConfig()
    processor = KosmosProcessor(config)
    
    try:
        with Image.open(image_path) as tiff_image:
            results = (seq(enumerate(ImageSequence.Iterator(tiff_image)))
                      .map(lambda page_data: processor.process_page(
                           page_data, task, num_beams, max_new_tokens, temperature))
                      .to_list())
            
            combined_output = (seq(results)
                             .map(lambda r: f"\n=== Page {r.page_num} ===\n{r.text}\n")
                             .reduce(lambda x, y: x + y))
            
            return results[0].image, combined_output
    except Exception as e:
        return None, f"Error processing multi-page TIFF: {str(e)}"

# Gradio interface remains the same
iface = gr.Interface(
    fn=process_multipage_tiff,
    inputs=[
        gr.Image(type="filepath", label="Input TIFF Image (Single or Multi-page)"),
        gr.Radio(["OCR", "Markdown"], label="Task", value="OCR"),
        gr.Slider(1, 10, value=4, step=1, label="Number of Beams"),
        gr.Slider(100, 4000, value=2048, step=100, label="Max New Tokens"),
        gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Temperature"),
    ],
    outputs=[
        gr.Image(type="pil", label="Image with Bounding Boxes (First page only)"),
        gr.Textbox(label="Extracted Text / Markdown (All pages)"),
    ],
    title="Kosmos 2.5 OCR and Markdown Generator (Multi-page TIFF Support)",
    description="""Generate OCR results or Markdown from single or multi-page TIFF images using Kosmos 2.5.""",
)