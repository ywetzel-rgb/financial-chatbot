from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from typing import List, Optional

app = FastAPI(title="Financial Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    client_id: str
    isin: Optional[str] = None
    fund_name: Optional[str] = None
    product_data: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str

CLIENT_CONFIGS = {
    "comdirect": {
        "name": "comdirect",
        "disclaimer": "Dies ist keine Anlageberatung von comdirect."
    },
    "consorsbank": {
        "name": "Consorsbank",
        "disclaimer": "Dies ist keine Anlageberatung von Consorsbank."
    },
    "default": {
        "name": "Finanzportal",
        "disclaimer": "Dies ist keine Anlageberatung."
    }
}

def get_system_prompt(client_id: str, isin: str = None, fund_name: str = None, product_data: dict = None):
    client_config = CLIENT_CONFIGS.get(client_id, CLIENT_CONFIGS["default"])
    
    base_prompt = f"""Du bist ein hilfreicher Assistent für Finanzprodukte bei {client_config['name']}.

STRIKTE REGELN - NIEMALS BRECHEN:
1. Du darfst KEINE Anlageberatung geben
2. Du darfst KEINE Kauf- oder Verkaufsempfehlungen aussprechen
3. Du darfst KEINE Renditeprognosen oder Kursziele nennen
4. Du darfst NICHT sagen "dieses Produkt ist besser als jenes"
5. Du darfst KEINE persönlichen Anlagestrategien empfehlen

WAS DU DARFST:
- Fachbegriffe erklären (z.B. "Was ist eine TER?", "Was bedeutet Tracking Error?")
- Allgemeine Funktionsweise von ETFs/Fonds erklären
- Unterschiede zwischen Anlageklassen erklären (Aktien vs. Anleihen)
- Allgemeine Informationen über Indizes geben (z.B. "Der MSCI World umfasst...")
- Risiken von Anlageklassen allgemein erklären
- Die aktuellen Produktdaten nennen, die dir zur Verfügung stehen

PRODUKTSPEZIFISCHE FRAGEN:
- Nutze dein Trainingswissen über bekannte ETFs und Fonds
- Bei unbekannten Produkten: Erkläre die Anlageklasse allgemein
- Wenn Produktdaten verfügbar sind, nutze diese für präzise Antworten

ANTWORTSTIL:
- Kurz und präzise (max. 3-4 Sätze)
- Professionell aber verständlich
- Keine Fachbegriffe ohne Erklärung
- Freundlich und hilfsbereit

DISCLAIMER:
Beende komplexe Antworten mit: "{client_config['disclaimer']}"
"""
    
    # ISIN und Fondsname hinzufügen
    if isin or fund_name:
        context = f"\n\nKONTEXT: Der Nutzer betrachtet gerade das Produkt:\n"
        if fund_name:
            context += f"- Name: {fund_name}\n"
        if isin:
            context += f"- ISIN: {isin}\n"
        base_prompt += context
    
    # NEU: Produktdaten hinzufügen
    if product_data:
        data_context = "\n\nAKTUELLE PRODUKTDATEN (verwende diese für präzise Antworten):\n"
        for key, value in product_data.items():
            data_context += f"- {key}: {value}\n"
        base_prompt += data_context
    
    return base_prompt

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.client_id:
            raise HTTPException(status_code=400, detail="client_id ist erforderlich")
        
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="Mindestens eine Nachricht erforderlich")
        
        system_prompt = get_system_prompt(
            client_id=request.client_id,
            isin=request.isin,
            fund_name=request.fund_name
            product_data=request.product_data
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        assistant_message = response.choices[0].message.content
        
        return ChatResponse(response=assistant_message)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ein Fehler ist aufgetreten: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "financial-chatbot-api",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "message": "Financial Chatbot API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
