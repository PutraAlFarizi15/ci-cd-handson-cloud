from fastapi import FastAPI
import os
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi.responses import HTMLResponse
import uvicorn

# Ambil port dari environment variable atau default ke 8080
port = int(os.environ.get("PORT", 8080))

# Inisialisasi model GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = FastAPI()

# Model Pydantic untuk menerima input dari pengguna
class TextGenerationRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def get_html():
    # Menyajikan file HTML untuk frontend
    try:
        html_file_path = os.path.join(os.path.dirname(__file__), "app/index.html")
        with open(html_file_path, "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"Error: {e}", status_code=500)

@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    prompt = request.prompt

    # Tokenisasi prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate teks
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode hasil output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)
