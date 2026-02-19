import os
import json
import re
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# 🔥 Libera CORS (necessário para InfinityFree)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

# Modelo mais estável no plano grátis
MODEL_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"


# ==============================
# FUNÇÃO AUXILIAR PARA LIMPAR JSON
# ==============================

def limpar_json(texto):
    try:
        clean = re.sub(r"```json|```", "", texto).strip()
        return json.loads(clean)
    except:
        return None


# ==============================
# ROTA RAIZ
# ==============================

@app.get("/")
def home():
    return {"status": "BrainUp API Online 🚀"}


# ==============================
# GERAR QUESTÃO
# ==============================

@app.get("/gerar")
async def gerar_questao(tema: str):

    if not HF_TOKEN:
        return JSONResponse(
            status_code=500,
            content={"erro": "HF_TOKEN não configurado no Render"}
        )

    prompt = f"""
Crie uma questão de múltipla escolha sobre {tema}.
Responda APENAS em JSON puro no formato:
{{
 "pergunta": "...",
 "opcoes": ["A) ...", "B) ...", "C) ...", "D) ..."],
 "correta": "A"
}}
"""

    try:
        response = requests.post(
            MODEL_URL,
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
            },
            timeout=60
        )

    except requests.exceptions.RequestException as e:
        return JSONResponse(
            status_code=500,
            content={"erro": "Erro ao conectar na Hugging Face", "detalhe": str(e)}
        )

    # 🔥 Se o modelo estiver carregando ou erro
    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={
                "erro": "Modelo indisponível",
                "status_code": response.status_code,
                "resposta": response.text
            }
        )

    try:
        result = response.json()
    except:
        return JSONResponse(
            status_code=500,
            content={"erro": "Resposta da IA não é JSON válido"}
        )

    try:
        texto = result[0]["generated_text"]
    except:
        return JSONResponse(
            status_code=500,
            content={"erro": "Formato inesperado da resposta da IA", "resposta": result}
        )

    questao = limpar_json(texto)

    if not questao:
        return JSONResponse(
            status_code=500,
            content={
                "erro": "IA não retornou JSON válido",
                "resposta_bruta": texto
            }
        )

    return questao


# ==============================
# ANALISAR IMAGEM (BRAINUP VISION)
# ==============================

@app.post("/analisar-imagem")
async def analisar_imagem(file: UploadFile = File(...)):

    if not HF_TOKEN:
        return JSONResponse(
            status_code=500,
            content={"erro": "HF_TOKEN não configurado"}
        )

    img_bytes = await file.read()

    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            },
            data=img_bytes,
            timeout=60
        )

    except requests.exceptions.RequestException as e:
        return JSONResponse(
            status_code=500,
            content={"erro": "Erro ao conectar no modelo de imagem", "detalhe": str(e)}
        )

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={
                "erro": "Modelo de imagem indisponível",
                "resposta": response.text
            }
        )

    try:
        result = response.json()
        descricao = result[0]["generated_text"]
    except:
        return JSONResponse(
            status_code=500,
            content={"erro": "Falha ao interpretar resposta da imagem"}
        )

    # Extrai possíveis temas da descrição
    temas = re.split(r",|\n|-", descricao)
    temas = [t.strip().capitalize() for t in temas if len(t.strip()) > 3]

    return {"temas": temas[:6]}
