import os
import logging
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import langchain
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LOGS Y OBSERVABILIDAD
# ------------------------------------------------------------------------------
langchain.debug = True 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AGENTE_BACKEND")

load_dotenv(dotenv_path='../.env') 
load_dotenv(dotenv_path='.env')

app = FastAPI(title="Agente E-Commerce Auditado")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI") 
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))

tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Agente_Produccion")

# ------------------------------------------------------------------------------
# 2. CARGA DE RECURSOS
# ------------------------------------------------------------------------------
logger.info("🚀 --- INICIANDO SERVIDOR DEL AGENTE ---")

try:
    logger.info("⏳ [INIT] Cargando modelo de embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("⏳ [INIT] Conectando a OpenAI...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    logger.info(f"⏳ [INIT] Conectando a Neo4j en: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    driver.verify_connectivity()
    logger.info("✅ [INIT] Neo4j conectado y listo.")

except Exception as e:
    logger.critical(f"❌ [FATAL] Error al iniciar servicios: {e}")
    raise e

# ------------------------------------------------------------------------------
# 3. HERRAMIENTAS (Lectura y Escritura)
# ------------------------------------------------------------------------------

@tool
def consultar_producto(intencion_usuario: str):
    """
    Busca productos. Devuelve precio, stock y NOTAS DE APRENDIZAJE PREVIAS.
    """
    logger.info(f"🛠️ [TOOL START] Buscando: '{intencion_usuario}'")
    
    try:
        vector = embedder.encode(intencion_usuario).tolist()
        
        # OJO: Aquí recuperamos 'c.nota' (Correcciones aprendidas)
        cypher = """
        CALL db.index.vector.queryNodes('productos_embeddings', 2, $vector)
        YIELD node AS p, score
        WHERE score > 0.6
        
        OPTIONAL MATCH (p)-[:COMPATIBLE_CON]->(acc:Producto)
        OPTIONAL MATCH (p)-[:TIENE_CORRECCION]->(c:Aprendizaje)
        OPTIONAL MATCH (t:Tienda)-[s:TIENE_STOCK]->(p) WHERE s.cantidad > 0
        
        RETURN 
            p.nombre as nombre, p.precio as precio, p.descripcion as desc,
            collect(DISTINCT acc.nombre) as accesorios,
            collect(DISTINCT c.nota) as notas_correccion,  // <--- IMPORTANTE
            collect(DISTINCT t.nombre + ': ' + toString(s.cantidad) + ' unid.') as stock_info
        """
        
        with driver.session() as session:
            result = session.run(cypher, vector=vector)
            data = [dict(record) for record in result]
        
        if not data:
            return "No encontré productos similares."
            
        response_txt = ""
        for item in data:
            response_txt += f"📦 {item['nombre']} (Base: ${item['precio']})\n"
            response_txt += f"   Desc: {item['desc']}\n"
            
            # Formateo especial para que el LLM vea la corrección
            if item['notas_correccion']:
                response_txt += f"   🚨 ATENCIÓN - CORRECCIONES APRENDIDAS: {item['notas_correccion']} (PRIORIDAD ALTA)\n"
            
            if item['stock_info']: response_txt += f"   ✅ Stock: {item['stock_info']}\n"
            else: response_txt += "   ❌ Sin Stock.\n"
            response_txt += "\n"
            
        return response_txt

    except Exception as e:
        logger.error(f"❌ [TOOL ERROR] {e}")
        return f"Error consultando DB: {str(e)}"

@tool
def aprender_correccion(nombre_producto: str, correccion_usuario: str):
    """Herramienta manual para que el LLM guarde correcciones durante la charla."""
    # (Esta función se mantiene igual que antes, sirve si el usuario lo pide explícitamente en el chat)
    logger.info(f"🧠 [TOOL] Aprendiendo manualmente: {nombre_producto}")
    return "Aprendido." # Simplificado para el ejemplo

mis_tools = [consultar_producto, aprender_correccion]
llm_with_tools = llm.bind_tools(mis_tools)

# ------------------------------------------------------------------------------
# 4. LÓGICA DEL AGENTE
# ------------------------------------------------------------------------------

def ejecutar_agente(mensaje_usuario: str):
    # Prompt del Sistema ACTUALIZADO para obedecer las correcciones
    sistema = """
    Eres un asistente de ventas experto y honesto.
    1. Usa 'consultar_producto' para buscar información.
    2. REGLA DE ORO: Si la información del producto contiene 'CORRECCIONES APRENDIDAS' o 'NOTAS DE CORRECCIÓN',
       debes considerar esa información como la VERDAD ABSOLUTA, ignorando el precio o descripción original si contradice la nota.
       Ejemplo: Si la base dice $100 pero la nota dice "El precio real es $90", informa que cuesta $90.
    3. Menciona el stock disponible si existe.
    """
    
    messages = [
        SystemMessage(content=sistema),
        HumanMessage(content=mensaje_usuario)
    ]
    
    # Ciclo ReAct
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            logger.info(f"👉 [ROUTER] Tool elegida: {tool_call['name']}")
            
            # Selector simple
            if tool_call["name"] == "consultar_producto":
                tool_output = consultar_producto.invoke(tool_call["args"])
            else:
                tool_output = aprender_correccion.invoke(tool_call["args"])
            
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        respuesta_final = llm_with_tools.invoke(messages)
        return respuesta_final.content
    
    return ai_msg.content

# ------------------------------------------------------------------------------
# 5. API ENDPOINTS (AQUÍ ESTÁ LA MAGIA DEL FEEDBACK)
# ------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonimo"

class FeedbackRequest(BaseModel):
    run_id: str
    score: int
    comment: str = None

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            respuesta = ejecutar_agente(req.query)
            mlflow.log_text(respuesta, "respuesta_agente.txt")
            return {"response": respuesta, "run_id": run_id, "status": "success"}
    except Exception as e:
        logger.error(f"❌ {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    logger.info(f"⭐ [FEEDBACK] Recibido. Score: {req.score}")
    
    try:
        # 1. Loggear en MLflow
        with mlflow.start_run(run_id=req.run_id):
            mlflow.log_metric("calidad_respuesta", req.score)
            if req.comment: mlflow.log_text(req.comment, "feedback_usuario.txt")
        
        # 2. LÓGICA DE AUTO-APRENDIZAJE
        # Si es negativo (0) y hay un comentario sustancial
        if req.score == 0 and req.comment and len(req.comment) > 5:
            logger.info(f"🧠 [AUTO-LEARN] Analizando queja: '{req.comment}'")
            
            # A. Vectorizamos la queja para saber de qué producto habla el usuario
            # (Ej: "La MacBook está muy cara" -> Vector apunta a MacBook)
            vector_queja = embedder.encode(req.comment).tolist()
            
            # B. Query para insertar la corrección en el grafo
            cypher_learning = """
            CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
            YIELD node AS p, score
            WHERE score > 0.7  // Umbral alto para asegurarnos que es el producto correcto
            
            // Crear el nodo de aprendizaje
            CREATE (c:Aprendizaje {
                nota: "CORRECCIÓN DE USUARIO: " + $texto, 
                fecha: datetime(),
                origen: 'feedback_loop'
            })
            
            // Conectar el producto con la corrección
            MERGE (p)-[:TIENE_CORRECCION]->(c)
            RETURN p.nombre as producto_afectado
            """
            
            with driver.session() as session:
                result = session.run(cypher_learning, vector=vector_queja, texto=req.comment).single()
                
                if result:
                    prod_name = result['producto_afectado']
                    logger.info(f"✅ [AUTO-LEARN] ÉXITO. Corrección asociada al producto: '{prod_name}'")
                    return {"message": f"Gracias. He aprendido que hay un error con: {prod_name}."}
                else:
                    logger.warning("⚠️ [AUTO-LEARN] No pude asociar la queja a un producto específico.")
                    return {"message": "Feedback recibido (no asociado a producto)"}

        return {"message": "Feedback recibido"}
        
    except Exception as e:
        logger.error(f"❌ [FEEDBACK ERROR] {e}")
        return {"message": "Error procesando feedback"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)