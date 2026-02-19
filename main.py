import os
import json
import re
import requests
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

HF_TOKEN = os.getenv("HF_TOKEN")

# ----------------------------
# Função para limpar JSON
# ----------------------------
def limpar_json(texto):
    clean = re.sub(r'```json|```', '', texto).strip()
    return json.loads(clean)

# ----------------------------
# GERAR QUESTÃO (Texto)
# ----------------------------
@app.get("/gerar")
async def gerar_questao(tema: str):

    prompt = f"""
Crie uma questão de múltipla escolha sobre {tema}.
Responda APENAS em JSON puro no formato:
{{
 "pergunta": "...",
 "opcoes": ["A) ...", "B) ...", "C) ...", "D) ..."],
 "correta": "A"
}}
"""

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7
            }
        }
    )

    result = response.json()

    try:
        texto = result[0]["generated_text"]
        return limpar_json(texto)
    except:
        return {"erro": "Falha ao gerar questão", "resposta_bruta": result}


# ----------------------------
# ANALISAR IMAGEM
# ----------------------------
@app.post("/analisar-imagem")
async def analisar_imagem(file: UploadFile = File(...)):

    img_bytes = await file.read()

    response = requests.post(
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}"
        },
        data=img_bytes
    )

    result = response.json()

    try:
        descricao = result[0]["generated_text"]

        # Transformar descrição em temas simples
        temas = descricao.split(",")[:5]

        return {"temas": temas}
    except:
        return {"erro": "Falha ao analisar imagem", "resposta_bruta": result}
