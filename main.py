import os
import json
import re
from fastapi import FastAPI
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

@app.get("/gerar")
def gerar_questao(tema: str):
    prompt_sistema = (
        "Você é um gerador de questões acadêmicas. Responda APENAS em formato JSON puro, sem textos extras ou markdown. "
        "O formato deve ser: "
        '{"pergunta": "...", "opcoes": ["A) ...", "B) ...", "C) ...", "D) ..."], "correta": "A"}'
    )
    
    prompt_usuario = f"Gere uma questão de múltipla escolha sobre {tema}."

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=600,
            temperature=0.7
        )
        # Limpa possíveis blocos de código markdown do JSON
        content = response.choices[0].message.content
        clean_json = re.sub(r'```json|```', '', content).strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"error": "Falha ao formatar JSON", "details": str(e)}
