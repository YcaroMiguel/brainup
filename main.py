import os
import json
import re
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura o Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def limpar_json(texto):
    # Remove blocos de código markdown se existirem
    clean = re.sub(r'```json|```', '', texto).strip()
    return json.loads(clean)

@app.get("/gerar")
async def gerar_questao(tema: str):
    prompt = (
        f"Gere uma questão de múltipla escolha sobre: {tema}. "
        "Responda APENAS em JSON puro: "
        '{"pergunta": "...", "opcoes": ["A) ...", "B) ...", "C) ...", "D) ..."], "correta": "Letra"}'
    )
    response = model.generate_content(prompt)
    return limpar_json(response.text)

@app.post("/analisar-imagem")
async def analisar_imagem(file: UploadFile = File(...)):
    img_bytes = await file.read()
    
    # Prepara a imagem para o Gemini
    img_data = [{"mime_type": file.content_type, "data": img_bytes}]
    
    prompt = (
        "Analise esta imagem de um cronograma ou material de estudo. "
        "Extraia os principais temas de estudo e retorne APENAS um JSON: "
        '{"temas": ["Tema 1", "Tema 2", "Tema 3"]}'
    )

    response = model.generate_content([prompt, img_data[0]])
    return limpar_json(response.text)
