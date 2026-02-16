import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

app = FastAPI()

# Configuração de CORS para o InfinityFree
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Puxa o Token das variáveis de ambiente do Render
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

@app.get("/gerar")
def gerar_questao(tema: str):
    # Prompt de engenharia para formatação rigorosa
    prompt_sistema = (
        "Você é um professor especialista em criar questões de exames. "
        "Sua resposta deve conter APENAS a questão, 4 alternativas (A, B, C, D) e o gabarito no final. "
        "Use markdown para negrito e pule linhas entre as opções."
    )
    
    prompt_usuario = f"Crie uma questão de nível médio sobre o tema: {tema}. Idioma: Português."

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
        return {"questao": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
