from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Fast-API in Containe Apakah sudah bisa berubah otomatis coba lagi?"}