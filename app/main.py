from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi.responses import HTMLResponse

# Inisialisasi model dan tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = FastAPI()

# Model Pydantic untuk menerima input dari pengguna
class TextGenerationRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def get_html():
    # Menyajikan file HTML untuk frontend
    with open("app/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    prompt = request.prompt

    # Tokenisasi prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate teks
    outputs = model.generate(inputs, 
                             max_length=100,
                             temperature=0.7,  # Mengurangi nilai agar hasil lebih fokus
                             top_k=50,
                             top_p=0.9,  # Sampling lebih baik
                             no_repeat_ngram_size=2)

    # Decode hasil output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}
