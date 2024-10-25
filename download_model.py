from transformers import AutoModelForVision2Seq, AutoProcessor

# Define the model name and local directory
model_name = "microsoft/kosmos-2.5"
local_dir = "./model"

# Download and save the model
model = AutoModelForVision2Seq.from_pretrained(model_name)
model.save_pretrained(local_dir)

# Download and save the processor
processor = AutoProcessor.from_pretrained(model_name)
processor.save_pretrained(local_dir)

print(f"Model and processor saved to {local_dir}")
