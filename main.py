import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# =========================
# CORS (permite requisições do front)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois você pode restringir ao domínio do front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIGURAÇÃO HF
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Modelo recomendado (OpenAI-compatible)
MODEL = "deepseek-ai/DeepSeek-R1:novita"

# =========================
# ROTA RAIZ (teste)
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

    prompt = (
        f"Crie uma questão de múltipla escolha sobre '{tema}'. "
        "Forneça 4 alternativas (A, B, C, D) e indique qual é a correta. "
        "Responda apenas em formato legível para JSON, mas sem JSON estrito, ex.:\n"
        "Pergunta: ...\nAlternativas: A) ... B) ... C) ... D) ...\nCorreta: X"
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            return {
                "erro": "Erro na HuggingFace",
                "status_code": response.status_code,
                "resposta": response.text
            }

        data = response.json()
        # Extrair apenas o texto gerado pelo modelo
        texto = data["choices"][0]["message"]["content"]

        return {"resultado": texto}

    except Exception as e:
        return {"erro": str(e)}
