import os
import json
import re
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

# Função auxiliar para limpar a resposta da IA e garantir que seja um JSON válido
def limpar_json(texto):
    clean = re.sub(r'```json|```', '', texto).strip()
    return json.loads(clean)

@app.get("/gerar")
def gerar_questao(tema: str):
    prompt_sistema = (
        "Você é um gerador de questões acadêmicas. Responda APENAS em formato JSON puro. "
        "Formato: "
        '{"pergunta": "...", "opcoes": ["A) ...", "B) ...", "C) ...", "D) ..."], "correta": "Letra"}'
    )
    prompt_usuario = f"Gere uma questão de múltipla escolha sobre: {tema}."

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": prompt_usuario}],
        max_tokens=600
    )
    return limpar_json(response.choices[0].message.content)

@app.post("/analisar-imagem")
async def analisar_imagem(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    prompt = (
        "Analise esta imagem de um cronograma ou material de estudo. "
        "Extraia os principais temas de estudo e retorne APENAS um JSON no formato: "
        '{"temas": ["Tema 1", "Tema 2", "Tema 3"]}'
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=500
    )
    return limpar_json(response.choices[0].message.content)
