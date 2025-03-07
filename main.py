import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define the model name from Hugging Face
model_name = "codellama/CodeLlama-7b-Instruct-hf"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model... This may take some time.")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # automatically map to available devices (GPU/CPU)
    torch_dtype=torch.float16,   # use FP16 if supported
)

# Define the request body model
class CodeRequest(BaseModel):
    prompt: str

# Initialize the FastAPI app
app = FastAPI(title="Code Generation API with CodeLlama 7B Instruct")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI server is running"}

# POST endpoint to generate code based on a prompt
@app.post("/generate")
async def generate_code(req: CodeRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    
    # Tokenize the incoming prompt and move tensors to the model's device
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    
    # Generate new tokens; adjust parameters as needed
    outputs = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + 150,  # generate up to 150 new tokens
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1,
    )
    
    # Decode the generated tokens to text
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_code": generated_code}
