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


langchain.debug = True 
# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LOGS Y OBSERVABILIDAD
# ------------------------------------------------------------------------------

# Configuración del Logger (Para que salga bonito en Docker)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AGENTE_BACKEND")


# Cargar variables
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

# Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI") 
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))

# MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Agente_Produccion")

# ------------------------------------------------------------------------------
# 2. CARGA DE RECURSOS (CON LOGS DE INICIO)
# ------------------------------------------------------------------------------
logger.info("🚀 --- INICIANDO SERVIDOR DEL AGENTE ---")

try:
    logger.info("⏳ [INIT] Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ [INIT] Embeddings cargados.")

    logger.info("⏳ [INIT] Conectando a OpenAI...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    logger.info(f"⏳ [INIT] Conectando a Neo4j en: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    driver.verify_connectivity() # Prueba real de conexión
    logger.info("✅ [INIT] Neo4j conectado y listo.")

except Exception as e:
    logger.critical(f"❌ [FATAL] Error al iniciar servicios: {e}")
    raise e

# ------------------------------------------------------------------------------
# 3. HERRAMIENTAS (CON LOGS DE EJECUCIÓN)
# ------------------------------------------------------------------------------

@tool
def consultar_producto(intencion_usuario: str):
    """
    Busca productos en el catálogo.
    Devuelve: Precio, Descripción, STOCK EN TIENDAS y Correcciones aprendidas.
    """
    # Log cuando la herramienta es invocada
    logger.info(f"🛠️ [TOOL START] 'consultar_producto' invocada. Query: '{intencion_usuario}'")
    
    try:
        vector = embedder.encode(intencion_usuario).tolist()
        
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
            collect(DISTINCT c.nota) as notas_correccion,
            collect(DISTINCT t.nombre + ': ' + toString(s.cantidad) + ' unid.') as stock_info
        """
        
        with driver.session() as session:
            result = session.run(cypher, vector=vector)
            data = [dict(record) for record in result]
        
        logger.info(f"📊 [NEO4J RESULT] Se encontraron {len(data)} productos.")
        
        if not data:
            logger.warning("⚠️ [NEO4J] Búsqueda vacía.")
            return "No encontré productos similares en la base de datos."
            
        # Construcción de respuesta (simplificada para no llenar código)
        response_txt = ""
        for item in data:
            response_txt += f"📦 {item['nombre']} (${item['precio']})\n"
            if item['stock_info']: response_txt += f"   ✅ Stock: {item['stock_info']}\n"
            else: response_txt += "   ❌ Sin Stock.\n"
            if item['notas_correccion']: response_txt += f"   ⚠️ NOTA: {item['notas_correccion']}\n"
            response_txt += "\n"
            
        logger.info("✅ [TOOL END] Respuesta generada correctamente.")
        return response_txt

    except Exception as e:
        logger.error(f"❌ [TOOL ERROR] Falló la consulta a Neo4j: {e}")
        return f"Error consultando DB: {str(e)}"

@tool
def aprender_correccion(nombre_producto: str, correccion_usuario: str):
    """Úsala CUANDO EL USUARIO CORRIGE un dato."""
    logger.info(f"🧠 [TOOL START] 'aprender_correccion'. Prod: {nombre_producto}, Nota: {correccion_usuario}")
    
    try:
        vector = embedder.encode(nombre_producto).tolist()
        cypher = """
        CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
        YIELD node AS p, score WHERE score > 0.6
        CREATE (c:Aprendizaje {nota: $nota, fecha: datetime()})
        MERGE (p)-[:TIENE_CORRECCION]->(c)
        RETURN p.nombre as nombre
        """
        with driver.session() as session:
            res = session.run(cypher, vector=vector, nota=correccion_usuario).single()
            
        if res:
            logger.info(f"✅ [LEARNING] Aprendizaje guardado en nodo: {res['nombre']}")
            return f"¡Gracias! He actualizado mi memoria para '{res['nombre']}'."
        else:
            logger.warning("⚠️ [LEARNING] No se encontró el producto base.")
            return "No pude encontrar el producto original."
            
    except Exception as e:
        logger.error(f"❌ [LEARNING ERROR] {e}")
        return f"Error guardando corrección: {str(e)}"

# Vincular herramientas
mis_tools = [consultar_producto, aprender_correccion]
llm_with_tools = llm.bind_tools(mis_tools)

# ------------------------------------------------------------------------------
# 4. LÓGICA DEL AGENTE
# ------------------------------------------------------------------------------

def ejecutar_agente(mensaje_usuario: str):
    logger.info(f"👤 [USER REQUEST] '{mensaje_usuario}'")
    
    messages = [
        SystemMessage(content="Eres un asistente de ventas experto... (mismas instrucciones)"),
        HumanMessage(content=mensaje_usuario)
    ]
    
    # 1. Planner
    logger.info("🤔 [LLM] El modelo está pensando qué herramienta usar...")
    # GRACIAS A set_debug(True), AQUÍ VERÁS EL LOG INTERNO DE LANGCHAIN EN CONSOLA
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    # 2. Tool Execution
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            logger.info(f"👉 [ROUTER] Herramienta seleccionada: {tool_call['name']}")
            
            selected_func = {
                "consultar_producto": consultar_producto,
                "aprender_correccion": aprender_correccion
            }[tool_call["name"]]
            
            tool_output = selected_func.invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        # 3. Synthesizer
        logger.info("📝 [LLM] Generando respuesta final en lenguaje natural...")
        respuesta_final = llm_with_tools.invoke(messages)
        return respuesta_final.content
    
    logger.info("ℹ️ [LLM] Respuesta directa (sin herramientas).")
    return ai_msg.content

# ------------------------------------------------------------------------------
# 5. API ENDPOINTS
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
    logger.info(f"📡 [API] Nueva petición POST /chat - User: {req.user_id}")
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("query", req.query)
            
            respuesta = ejecutar_agente(req.query)
            
            mlflow.log_text(respuesta, "respuesta_agente.txt")
            return {"response": respuesta, "run_id": run_id, "status": "success"}
    except Exception as e:
        logger.error(f"❌ [API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    logger.info(f"⭐ [API] Feedback recibido. Score: {req.score}")
    try:
        with mlflow.start_run(run_id=req.run_id):
            mlflow.log_metric("calidad_respuesta", req.score)
            if req.comment: mlflow.log_text(req.comment, "feedback_usuario.txt")
        
        # Lógica de auto-aprendizaje
        if req.score == 0 and req.comment and len(req.comment) > 10:
            logger.info(f"🧠 [AUTO-LEARN] Intentando aprender de la queja: {req.comment}")
            # ... (Aquí va tu lógica de inyección automática del código anterior) ...
            # Para resumir el log, solo invoco embedder
            vector_queja = embedder.encode(req.comment).tolist()
            # ... (Lógica de escritura en Neo4j) ...
            logger.info("✅ [AUTO-LEARN] Proceso de aprendizaje completado.")

        return {"message": "Feedback recibido"}
    except Exception as e:
        logger.error(f"❌ [FEEDBACK ERROR] {e}")
        return {"message": "Feedback recibido (con errores internos)"}

if __name__ == "__main__":
    import uvicorn
    # Log para saber si estamos en local
    logger.info("🖥️ Iniciando servidor en modo LOCAL (fuera de Docker)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)