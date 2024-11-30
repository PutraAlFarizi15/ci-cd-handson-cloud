from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Inisialisasi model GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Inisialisasi FastAPI dan templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>GPT-2 Text Generator</title>
        </head>
        <body>
            <h1>GPT-2 Text Generator</h1>
            <form method="post" action="/generate/">
                <textarea name="prompt" rows="4" cols="50" placeholder="Enter your prompt here..." required></textarea><br><br>
                <button type="submit">Generate</button>
            </form>
        </body>
    </html>
    """

@app.post("/generate/", response_class=HTMLResponse)
async def generate_text(prompt: str = Form(...)):
    # Tokenisasi input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Menghasilkan teks
    output = model.generate(
    input_ids=input_ids,
    max_length=150,
    temperature=0.7,  # Mengurangi nilai agar hasil lebih fokus
    top_k=50,
    top_p=0.9,  # Sampling lebih baik
    no_repeat_ngram_size=2
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return f"""
    <html>
        <head>
            <title>GPT-2 Text Generator</title>
        </head>
        <body>
            <h1>GPT-2 Text Generator</h1>
            <form method="post" action="/generate/">
                <textarea name="prompt" rows="4" cols="50" placeholder="Enter your prompt here..." required></textarea><br><br>
                <button type="submit">Generate</button>
            </form>
            <h2>Your Prompt:</h2>
            <p>{prompt}</p>
            <h2>Generated Text:</h2>
            <p>{generated_text}</p>
        </body>
    </html>
    """
