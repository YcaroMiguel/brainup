import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# =========================
# CORS (IMPORTANTE PRO FRONT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois você pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "google/flan-t5-base"

HF_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# =========================
# ROTA RAIZ (TESTE)
# =========================
@app.get("/")
async def home():
    return {"status": "BrainUp backend online 🚀"}

# =========================
# ROTA GERAR QUESTÃO
# =========================
@app.get("/gerar")
async def gerar_questao(tema: str):

    if not HF_TOKEN:
        return {"erro": "HF_TOKEN não configurado no Render"}

    prompt = f"""
Crie uma questão de múltipla escolha sobre {tema}.
Forneça 4 alternativas (A, B, C, D).
No final, indique qual é a correta.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(HF_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return {
                "erro": "Erro na HuggingFace",
                "status_code": response.status_code,
                "resposta": response.text
            }

        data = response.json()

        if isinstance(data, list):
            texto = data[0].get("generated_text", "")
        else:
            texto = data.get("generated_text", "")

        return {"resultado": texto}

    except Exception as e:
        return {"erro": str(e)}
