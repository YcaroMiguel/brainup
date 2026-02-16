from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os

app = FastAPI()

# Permite que o seu site no InfinityFree acesse esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

client = InferenceClient(api_key="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")

@app.get("/gerar")
def gerar_questao(tema: str):
    prompt = f"Crie uma questão de múltipla escolha sobre {tema} com gabarito ao final."
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return {"questao": response.choices[0].message.content}
