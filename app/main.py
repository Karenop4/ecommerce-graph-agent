import json
import re
import uuid
import time
import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import mlflow
import langchain
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sentence_transformers import SentenceTransformer

import redis

from core.config import load_settings
from core.logging_utils import setup_structured_logging
from core.guards import is_off_topic_query, off_topic_response
from core.graph_search import buscar_en_grafo, format_graph_search_results

# ----------------------------------------------------------------------
# 0) CONFIG
# ----------------------------------------------------------------------
langchain.debug = True
logger = setup_structured_logging()
settings = load_settings()


app = FastAPI(title="Agente E-Commerce Conversacional + Redis (Planner + Logs)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = settings.openai_api_key
NEO4J_URI = settings.neo4j_uri
NEO4J_AUTH = settings.neo4j_auth
REDIS_URL = settings.redis_url
REDIS_TTL_SECONDS = settings.redis_ttl_seconds
TOOLS_PRINT_ONLY = settings.tools_print_only

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment)

logger.info("config_loaded", extra={"extra": {"openai_api_key_loaded": bool(OPENAI_API_KEY), "neo4j_uri": NEO4J_URI, "redis_url": REDIS_URL}})

# Recursos globales inicializados en startup (no en import)
embedder: Optional[SentenceTransformer] = None
llm: Optional[ChatOpenAI] = None
driver = None
rdb = None
GRAPH_HIGHLIGHTS: Dict[str, List[str]] = {}
ORCHESTRATION_GRAPH = None


# ----------------------------------------------------------------------
# 1) RETRIES (Neo4j / Redis)
# ----------------------------------------------------------------------
def connect_neo4j_with_retry(uri: str, auth, attempts: int = 40, sleep_s: float = 2.0):
    """Conecta a Neo4j con reintentos para esperar a que el contenedor esté listo."""
    last_err = None
    for i in range(1, attempts + 1):
        try:
            d = GraphDatabase.driver(uri, auth=auth)
            d.verify_connectivity()
            logger.info(f"✅ [INIT] Neo4j listo (intento {i}/{attempts}) uri={uri}")
            return d
        except ServiceUnavailable as e:
            last_err = e
            logger.warning(f"⏳ [INIT] Neo4j no listo (intento {i}/{attempts}) -> {e}")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            logger.warning(f"⏳ [INIT] Error Neo4j (intento {i}/{attempts}) -> {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"❌ No se pudo conectar a Neo4j tras {attempts} intentos. Último error: {last_err}")


def connect_redis_with_retry(url: str, attempts: int = 40, sleep_s: float = 1.0):
    """Conecta a Redis con reintentos para esperar a que el contenedor esté listo."""
    last_err = None
    for i in range(1, attempts + 1):
        try:
            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            logger.info(f"✅ [INIT] Redis listo (intento {i}/{attempts}) url={url}")
            return client
        except Exception as e:
            last_err = e
            logger.warning(f"⏳ [INIT] Redis no listo (intento {i}/{attempts}) -> {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"❌ No se pudo conectar a Redis tras {attempts} intentos. Último error: {last_err}")


# ----------------------------------------------------------------------
# 2) ROUTER LABELS / EMBEDS (precomputados en startup)
# ----------------------------------------------------------------------
ROUTER_LABELS = [
    "browse_products",
    "choose_option",
    "add_to_cart",
    "remove_from_cart",
    "view_cart",
    "purchase_or_stock",
    "store_contact",
    "finalize_purchase",
    "user_correction",
    "respuesta_directa",
]
ROUTER_EMBEDS: Optional[List[List[float]]] = None
DIRECT_RESPONSE_MARGIN = 0.06

INTENT_KEYWORDS = {
    "browse_products": [r"\bbusco\b", r"\brecomiend", r"\bquiero ver\b", r"\bmostrar productos\b"],
    "choose_option": [r"\bopci[oó]n\s*\d+", r"\bla segunda\b", r"\bla tercera\b", r"\bla primera\b"],
    "add_to_cart": [r"\bagrega\b", r"\bañade\b", r"\bme llevo\b", r"\bponer en carrito\b"],
    "remove_from_cart": [r"\bquita\b", r"\bremueve\b", r"\bsacar del carrito\b"],
    "view_cart": [r"\bver carrito\b", r"\bmi carrito\b", r"\bque tengo\b"],
    "purchase_or_stock": [r"\bstock\b", r"\bdisponibilidad\b", r"\bcomprar\b"],
    "store_contact": [r"\bwhatsapp\b", r"\btel[eé]fono\b", r"\bdirecci[oó]n\b", r"\bhorario\b"],
    "finalize_purchase": [r"\bfinalizar compra\b", r"\bproceder a compra\b", r"\bhaz la compra\b"],
    "user_correction": [r"\best[aá] mal\b", r"\bcorrige\b", r"\bno es as[ií]\b"],
}


def cosine_sim(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def seleccionar_funcion(query_text: str, query_vec_norm: List[float]) -> Tuple[str, float, List[float], str]:
    """Devuelve label + score + sims + strategy para debugging."""
    normalized = (query_text or "").strip().lower()

    # 1) Regla por intención explícita (prioridad alta)
    for label, patterns in INTENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                sims = [0.0 for _ in ROUTER_LABELS]
                if label in ROUTER_LABELS:
                    sims[ROUTER_LABELS.index(label)] = 1.0
                return label, 1.0, sims, "keyword"

    # 2) Router semántico por similitud coseno
    sims = [cosine_sim(query_vec_norm, ROUTER_EMBEDS[i]) for i in range(len(ROUTER_LABELS))]

    # Penaliza respuesta_directa como fallback para evitar sesgo
    adjusted_sims = list(sims)
    if "respuesta_directa" in ROUTER_LABELS:
        idx_direct = ROUTER_LABELS.index("respuesta_directa")
        adjusted_sims[idx_direct] = adjusted_sims[idx_direct] - 0.08

    best_idx = max(range(len(adjusted_sims)), key=lambda i: adjusted_sims[i])

    # Si respuesta_directa no gana con margen claro, preferimos el siguiente intent
    if ROUTER_LABELS[best_idx] == "respuesta_directa":
        ranked = sorted(
            [(i, float(adjusted_sims[i])) for i in range(len(adjusted_sims))],
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ranked) > 1:
            _, best_score = ranked[0]
            second_idx, second_score = ranked[1]
            if (best_score - second_score) <= DIRECT_RESPONSE_MARGIN:
                return ROUTER_LABELS[second_idx], second_score, adjusted_sims, "semantic_margin"

    return ROUTER_LABELS[best_idx], float(adjusted_sims[best_idx]), adjusted_sims, "semantic"


@app.on_event("startup")
def on_startup():
    """Inicializa recursos pesados en startup (mejor para Docker y --reload)."""
    global embedder, llm, driver, rdb, ROUTER_EMBEDS, ORCHESTRATION_GRAPH

    logger.info("🚀 --- STARTUP ---")

    logger.info("⏳ [INIT] Cargando embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✅ [INIT] Embeddings OK")

    logger.info("⏳ [INIT] Conectando a OpenAI (client)...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    logger.info("✅ [INIT] LLM OK")

    logger.info(f"⏳ [INIT] Conectando a Neo4j: {NEO4J_URI}")
    driver = connect_neo4j_with_retry(NEO4J_URI, NEO4J_AUTH, attempts=40, sleep_s=2.0)

    logger.info(f"⏳ [INIT] Conectando a Redis: {REDIS_URL}")
    rdb = connect_redis_with_retry(REDIS_URL, attempts=40, sleep_s=1.0)

    # Router embeds una sola vez
    ROUTER_EMBEDS = embedder.encode(ROUTER_LABELS, normalize_embeddings=True).tolist()
    logger.info("✅ [INIT] Router embeds precalculados")

    ORCHESTRATION_GRAPH = build_orchestration_graph()
    logger.info("✅ [INIT] Orquestación LangGraph lista")

    if "localhost" in (REDIS_URL or ""):
        logger.warning("⚠️ REDIS_URL usa localhost. En Docker debe ser redis://redis:6379/0 (nombre del servicio).")


@app.on_event("shutdown")
def on_shutdown():
    global driver
    if driver:
        driver.close()
        logger.info("🛑 Neo4j driver closed")


# ----------------------------------------------------------------------
# 3) REDIS STATE
# ----------------------------------------------------------------------
def _state_key(user_id: str) -> str:
    return f"session:{user_id}"


DEFAULT_STATE = {
    "stage": "explore",              # explore | decide | buy | contact | done
    "selected_product_id": None,     # “focus” actual
    "selected_store": None,
    "last_candidates": [],           # [{id,nombre,precio}]
    "last_intent": None,
    "cart_items": [],                # [{id,nombre,precio,qty}]
}


def get_state(user_id: str) -> Dict[str, Any]:
    """Lee estado del usuario desde Redis."""
    raw = rdb.get(_state_key(user_id)) if rdb else None
    if not raw:
        return dict(DEFAULT_STATE)
    try:
        s = json.loads(raw)
        if not isinstance(s, dict):
            return dict(DEFAULT_STATE)
        for k, v in DEFAULT_STATE.items():
            if k not in s:
                s[k] = v
        if not isinstance(s.get("cart_items"), list):
            s["cart_items"] = []
        if not isinstance(s.get("last_candidates"), list):
            s["last_candidates"] = []
        return s
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(user_id: str, state: Dict[str, Any]) -> None:
    """Guarda estado del usuario en Redis con TTL."""
    if not rdb:
        return
    rdb.setex(_state_key(user_id), REDIS_TTL_SECONDS, json.dumps(state, ensure_ascii=False))


def update_state(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Actualiza estado (merge) y guarda."""
    state = get_state(user_id)
    state.update(patch)
    save_state(user_id, state)
    return state


def _queue_key(user_id: str) -> str:
    return f"task_queue:{user_id}"


def _events_key(user_id: str) -> str:
    return f"events:{user_id}"


def _sanitize_event_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def strip_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: strip_value(v)
                for k, v in value.items()
                if k not in ("user_id", "trace_id", "timestamp", "ts")
            }
        if isinstance(value, list):
            return [strip_value(v) for v in value]
        return value

    return strip_value(payload or {})


def emit_event(user_id: str, level: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    if not rdb:
        return
    data = {
        "level": level,
        "event": event,
        "payload": _sanitize_event_payload(payload),
    }
    rdb.rpush(_events_key(user_id), json.dumps(data, ensure_ascii=False))
    rdb.expire(_events_key(user_id), REDIS_TTL_SECONDS)


def get_pending_queue(user_id: str) -> Optional[Dict[str, Any]]:
    if not rdb:
        return None
    raw = rdb.get(_queue_key(user_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            return data
    except Exception:
        return None
    return None


def save_queue(user_id: str, queue_data: Dict[str, Any]) -> None:
    if not rdb:
        return
    rdb.setex(_queue_key(user_id), REDIS_TTL_SECONDS, json.dumps(queue_data, ensure_ascii=False))


def clear_queue(user_id: str) -> None:
    if rdb:
        rdb.delete(_queue_key(user_id))


def emit_graph_highlight_for_tool(user_id: str, tool_name: str, args: Dict[str, Any]) -> None:
    """Emite highlights de grafo para cualquier tool (productos/tiendas/carrito)."""
    if not driver:
        return

    state = get_state(user_id)
    product_ids = set()
    store_names = set()

    selected_product_id = state.get("selected_product_id")
    if selected_product_id:
        product_ids.add(str(selected_product_id))

    selected_store = state.get("selected_store")
    if selected_store:
        store_names.add(str(selected_store))

    for item in state.get("cart_items", []) or []:
        pid = item.get("id")
        if pid:
            product_ids.add(str(pid))

    for item in state.get("last_candidates", []) or []:
        pid = item.get("id")
        if pid:
            product_ids.add(str(pid))

    for key in ("producto_ref", "id", "product_id"):
        val = args.get(key)
        if isinstance(val, str) and re.match(r"^[A-Za-z]\d+$", val.strip()):
            product_ids.add(val.strip())

    tienda_arg = args.get("tienda") or args.get("nombre_tienda")
    if isinstance(tienda_arg, str) and tienda_arg.strip():
        store_names.add(tienda_arg.strip())

    if not product_ids and not store_names:
        return

    cypher_products = """
    MATCH (p:Producto)
    WHERE p.id IN $product_ids
    OPTIONAL MATCH (p)-[rp]-(pn)
    RETURN
      collect(DISTINCT elementId(p)) + collect(DISTINCT elementId(pn)) AS node_ids,
      collect(DISTINCT {from: elementId(p), to: elementId(pn), type: type(rp)}) AS edges
    """

    cypher_stores = """
    MATCH (t:Tienda)
    WHERE toLower(t.nombre) IN $store_names_lower
    OPTIONAL MATCH (t)-[rt]-(tn)
    RETURN
      collect(DISTINCT elementId(t)) + collect(DISTINCT elementId(tn)) AS node_ids,
      collect(DISTINCT {from: elementId(t), to: elementId(tn), type: type(rt)}) AS edges
    """

    store_names_lower = [s.lower() for s in store_names]

    try:
        node_ids: List[str] = []
        edges: List[Dict[str, Any]] = []
        with driver.session() as session:
            if product_ids:
                row_products = session.run(
                    cypher_products,
                    product_ids=list(product_ids),
                ).single()
                if row_products:
                    node_ids.extend(row_products.get("node_ids") or [])
                    edges.extend(row_products.get("edges") or [])

            if store_names_lower:
                row_stores = session.run(
                    cypher_stores,
                    store_names_lower=store_names_lower,
                ).single()
                if row_stores:
                    node_ids.extend(row_stores.get("node_ids") or [])
                    edges.extend(row_stores.get("edges") or [])

        node_ids = list(dict.fromkeys([n for n in node_ids if n]))
        edge_seen = set()
        dedup_edges = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            from_id = e.get("from")
            to_id = e.get("to")
            if not from_id or not to_id:
                continue
            key = f"{from_id}|{to_id}|{e.get('type') or ''}"
            if key in edge_seen:
                continue
            edge_seen.add(key)
            dedup_edges.append({"from": from_id, "to": to_id, "type": e.get("type")})

        if not node_ids:
            return

        GRAPH_HIGHLIGHTS[user_id] = node_ids
        emit_event(user_id, "info", "graph_search_complete", {
            "source_tool": tool_name,
            "node_ids": node_ids,
            "edges": dedup_edges,
            "total_depth": 1,
        })
    except Exception as e:
        logger.warning(f"[GRAPH][HIGHLIGHT] no se pudo emitir highlight para {tool_name}: {e}")


def normalize_ordinal_to_index(text: str) -> Optional[int]:
    """Convierte 'la segunda/2/tercera' a índice 0-based si aplica."""
    t = (text or "").lower().strip()
    mapping = {
        "1": 0, "primera": 0, "primer": 0, "primero": 0, "la primera": 0, "el primero": 0,
        "2": 1, "segunda": 1, "segundo": 1, "la segunda": 1, "el segundo": 1,
        "3": 2, "tercera": 2, "tercero": 2, "la tercera": 2, "el tercero": 2,
        "4": 3, "cuarta": 3, "cuarto": 3, "la cuarta": 3, "el cuarto": 3,
        "5": 4, "quinta": 4, "quinto": 4, "la quinta": 4, "el quinto": 4,
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return None


def _safe_int(x, default=1) -> int:
    """Convierte a int de forma segura."""
    try:
        return int(x)
    except Exception:
        return default


# -------------------------
# CARRITO helpers
# -------------------------
def cart_add_item(state: Dict[str, Any], item: Dict[str, Any], qty: int = 1) -> Dict[str, Any]:
    """Agrega (o suma qty) de un item al carrito en el state."""
    cart = state.get("cart_items", []) or []
    pid = item.get("id")
    if not pid:
        return state

    qty = max(1, _safe_int(qty, 1))

    for c in cart:
        if c.get("id") == pid:
            c["qty"] = _safe_int(c.get("qty", 1), 1) + qty
            state["cart_items"] = cart
            return state

    cart.append({
        "id": pid,
        "nombre": item.get("nombre"),
        "precio": item.get("precio"),
        "qty": qty,
    })
    state["cart_items"] = cart
    return state


def cart_remove_by_name_or_id(state: Dict[str, Any], item_ref: str) -> Tuple[Dict[str, Any], bool]:
    """Remueve un item del carrito por id o por match parcial del nombre."""
    cart = state.get("cart_items", []) or []
    ref = (item_ref or "").strip().lower()
    if not ref:
        return state, False

    new_cart = []
    removed = False
    for c in cart:
        cid = str(c.get("id", "")).lower()
        nombre = str(c.get("nombre", "")).lower()
        if ref == cid or ref in nombre:
            removed = True
            continue
        new_cart.append(c)

    state["cart_items"] = new_cart
    return state, removed


def cart_remove_by_indexes(state: Dict[str, Any], indexes_0based: List[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Remueve ítems del carrito por índices 0-based.
    Devuelve (state, removed_items).
    """
    cart = state.get("cart_items", []) or []
    if not cart:
        return state, []

    idx_set = set(i for i in indexes_0based if isinstance(i, int))
    removed_items = []
    new_cart = []
    for i, item in enumerate(cart):
        if i in idx_set:
            removed_items.append(item)
        else:
            new_cart.append(item)

    state["cart_items"] = new_cart
    return state, removed_items


def cart_clear(state: Dict[str, Any]) -> Dict[str, Any]:
    """Vacía el carrito."""
    state["cart_items"] = []
    return state


def cart_to_text(state: Dict[str, Any]) -> str:
    """Convierte carrito a texto amigable."""
    cart = state.get("cart_items", []) or []
    if not cart:
        return "🛒 Carrito vacío."
    total = 0
    txt = "🛒 Carrito:\n"
    for i, c in enumerate(cart, start=1):
        precio = c.get("precio") or 0
        qty = _safe_int(c.get("qty", 1), 1)
        subtotal = precio * qty
        total += subtotal
        txt += f"{i}) {c.get('nombre')} [{c.get('id')}] x{qty} = ${subtotal}\n"
    txt += f"Total estimado: ${total}\n"
    return txt


def parse_cart_indexes(text: str) -> List[int]:
    """
    Extrae índices del carrito a partir de:
    - "1", "2 y 3", "1,2,3"
    - "item 1 y 3", "items 2, 4"
    Devuelve índices 0-based.
    """
    raw = (text or "").lower()
    raw = raw.replace("items", "").replace("item", "")
    raw = raw.replace(" y ", ",")
    raw = raw.replace(";", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    idxs = []
    for p in parts:
        # intenta leer números
        try:
            n = int(p)
            if n > 0:
                idxs.append(n - 1)
        except Exception:
            continue
    return idxs


# ----------------------------------------------------------------------
# 4) TOOLS
# ----------------------------------------------------------------------
@tool
def buscar_productos(query: str, user_id: str = "anonimo") -> str:
    """Explora catálogo (NO stock). Guarda candidatos en Redis para usar 'la segunda'."""
    logger.info(f"🛠️ [TOOL] buscar_productos | query='{query}'")
    v = embedder.encode(query).tolist()

    cypher = """
    CALL db.index.vector.queryNodes('productos_embeddings', 5, $vector)
    YIELD node AS p, score
    WHERE score > 0.5

    OPTIONAL MATCH (p)-[:COMPATIBLE_CON]->(acc:Producto)
    OPTIONAL MATCH (p)-[:TIENE_CORRECCION]->(c:Aprendizaje)

    RETURN
      p.id AS id,
      p.nombre AS nombre,
      p.precio AS precio,
      p.descripcion AS desc,
      collect(DISTINCT acc.nombre) AS accesorios,
      collect(DISTINCT c.nota) AS correcciones
    """

    with driver.session() as session:
        rows = [dict(r) for r in session.run(cypher, vector=v)]

    if not rows:
        update_state(user_id, {"stage": "explore", "last_candidates": [], "selected_product_id": None})
        return "No se encontraron productos similares. ¿Buscas laptop, mouse, monitor o audífonos?"

    candidates = [{"id": r["id"], "nombre": r["nombre"], "precio": r["precio"]} for r in rows]
    update_state(user_id, {"stage": "decide", "last_candidates": candidates, "selected_product_id": candidates[0]["id"]})

    txt = "Opciones encontradas (aún sin ver stock):\n"
    for i, r in enumerate(rows, start=1):
        txt += f"{i}) [{r['id']}] {r['nombre']} (${r['precio']})\n"
        txt += f"   Desc: {r['desc']}\n"
        if r["correcciones"]:
            txt += f"   🚨 Correcciones aprendidas: {r['correcciones']}\n"
        if r["accesorios"]:
            txt += f"   💡 Accesorios: {', '.join(r['accesorios'])}\n"
        txt += "\n"
    return txt


@tool
def seleccionar_opcion(opcion: str, user_id: str = "anonimo") -> str:
    """Selecciona un producto por ordinal ('la segunda', '2') usando last_candidates."""
    logger.info(f"🛠️ [TOOL] seleccionar_opcion | opcion='{opcion}'")
    state = get_state(user_id)
    candidates = state.get("last_candidates", []) or []
    if not candidates:
        return "Aún no te he mostrado opciones. Dime qué estás buscando y te doy alternativas."

    idx = normalize_ordinal_to_index(opcion)
    if idx is None or idx < 0 or idx >= len(candidates):
        return "No entendí cuál opción. Puedes decir: 1, 2 o 3 (ej: 'la segunda')."

    chosen = candidates[idx]
    update_state(user_id, {"selected_product_id": chosen["id"], "stage": "decide"})
    return f"Perfecto ✅ seleccionaste: {chosen['nombre']} [{chosen['id']}]. ¿Quieres agregarlo al carrito o ver stock?"


@tool
def agregar_al_carrito(producto_ref: str, qty: int = 1, user_id: str = "anonimo") -> str:
    """
    Agrega producto al carrito.
    producto_ref puede ser ordinal ('la segunda'), id ('L3') o nombre ('Asus ROG...').
    """
    logger.info(f"🛠️ [TOOL] agregar_al_carrito | ref='{producto_ref}' qty={qty}")
    state = get_state(user_id)
    ref = (producto_ref or "").strip()
    if not ref:
        return "¿Qué producto agrego? (ej: 'la segunda', 'L3' o el nombre)."

    qty = max(1, _safe_int(qty, 1))

    # (1) ordinal con candidates
    idx = normalize_ordinal_to_index(ref)
    if idx is not None:
        candidates = state.get("last_candidates", []) or []
        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            state = cart_add_item(state, chosen, qty=qty)
            state["selected_product_id"] = chosen["id"]
            state["stage"] = "decide"
            save_state(user_id, state)
            return f"✅ Agregado: {chosen['nombre']} x{qty}\n\n{cart_to_text(state)}"
        return "No encontré esa opción en la lista. Dime 1, 2 o 3."

    # (2) resolver por id o embedding
    with driver.session() as session:
        prod = None
        looks_like_id = len(ref) <= 6 and ref[:1].isalpha()
        if looks_like_id:
            prod = session.run(
                "MATCH (p:Producto {id:$id}) RETURN p.id AS id, p.nombre AS nombre, p.precio AS precio LIMIT 1",
                id=ref.upper()
            ).single()
        if not prod:
            v = embedder.encode(ref).tolist()
            prod = session.run("""
                CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
                YIELD node AS p, score
                WHERE score > 0.6
                RETURN p.id AS id, p.nombre AS nombre, p.precio AS precio
            """, vector=v).single()

    if not prod:
        return "No pude identificar ese producto para agregar al carrito."

    item = {"id": prod["id"], "nombre": prod["nombre"], "precio": prod["precio"]}
    state = cart_add_item(state, item, qty=qty)
    state["selected_product_id"] = item["id"]
    state["stage"] = "decide"
    save_state(user_id, state)

    return f"✅ Agregado: {item['nombre']} x{qty}\n\n{cart_to_text(state)}"


@tool
def ver_carrito(user_id: str = "anonimo") -> str:
    """Muestra el carrito actual del usuario usando Redis."""
    logger.info("🛠️ [TOOL] ver_carrito")
    return cart_to_text(get_state(user_id))


@tool
def vaciar_carrito(user_id: str = "anonimo") -> str:
    """Vacía completamente el carrito (solo si el usuario lo pidió explícitamente)."""
    logger.info("🛠️ [TOOL] vaciar_carrito")
    state = get_state(user_id)
    state = cart_clear(state)
    save_state(user_id, state)
    return "🧹 Listo. Carrito vaciado."


@tool
def remover_del_carrito(items_ref: str, user_id: str = "anonimo") -> str:
    """
    Remueve UNO o VARIOS ítems del carrito.

    Soporta:
    - Por nombre/id: "logitech, razer y dell" / "L2"
    - Por índice: "1 y 3" / "item 2" / "items 1,3"
    """
    logger.info(f"🛠️ [TOOL] remover_del_carrito | items_ref='{items_ref}'")
    state = get_state(user_id)
    cart_before = list(state.get("cart_items", []) or [])
    if not cart_before:
        return "Tu carrito ya está vacío."

    raw = (items_ref or "").strip().lower()
    if not raw:
        return "¿Qué quieres quitar del carrito? (ej: 'quita el 1', 'quita logitech')."

    removed_any = False
    removed_list = []
    not_found = []

    # (A) Si el usuario metió números (por índice)
    idxs = parse_cart_indexes(raw)
    if idxs:
        state, removed_items = cart_remove_by_indexes(state, idxs)
        if removed_items:
            removed_any = True
            for it in removed_items:
                removed_list.append(f"{it.get('nombre')} [{it.get('id')}]")
        # índices inválidos (fuera de rango)
        max_idx = len(cart_before) - 1
        invalid = [i for i in idxs if i < 0 or i > max_idx]
        if invalid:
            not_found.append("índices: " + ", ".join(str(i + 1) for i in invalid))

        save_state(user_id, state)

        msg = ""
        if removed_any:
            msg += "✅ Quité del carrito: " + ", ".join(removed_list) + ".\n"
        if not_found:
            msg += "⚠️ No encontré: " + ", ".join(not_found) + ".\n"
        msg += "\n" + cart_to_text(state)
        return msg

    # (B) Split simple por comas y 'y' (por nombre o id)
    raw = raw.replace(" y ", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    for part in parts:
        state, removed = cart_remove_by_name_or_id(state, part)
        if removed:
            removed_any = True
            removed_list.append(part)
        else:
            not_found.append(part)

    save_state(user_id, state)

    msg = ""
    if removed_any:
        msg += f"✅ Quité del carrito: {', '.join(removed_list)}.\n"
    if not_found:
        msg += f"⚠️ No encontré en el carrito: {', '.join(not_found)}.\n"
    msg += "\n" + cart_to_text(state)
    return msg


@tool
def verificar_stock(producto_ref: str = "", tienda: str = "", user_id: str = "anonimo") -> str:
    """Revisa stock por tienda para un producto. Si producto_ref vacío, usa selected_product_id."""
    state = get_state(user_id)
    if not (producto_ref or "").strip():
        if state.get("selected_product_id"):
            producto_ref = state["selected_product_id"]
        else:
            return "¿De cuál producto? Puedes decir 'la primera/segunda' o escribir el nombre."

    logger.info(f"🛠️ [TOOL] verificar_stock | producto_ref='{producto_ref}' tienda='{tienda}'")

    with driver.session() as session:
        prod = None
        ref = producto_ref.strip()
        looks_like_id = len(ref) <= 6 and ref[:1].isalpha()
        if looks_like_id:
            prod = session.run(
                "MATCH (p:Producto {id:$id}) RETURN p.id AS id, p.nombre AS nombre LIMIT 1",
                id=ref.upper()
            ).single()
        if not prod:
            v = embedder.encode(ref).tolist()
            prod = session.run("""
                CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
                YIELD node AS p, score
                WHERE score > 0.6
                RETURN p.id AS id, p.nombre AS nombre
            """, vector=v).single()

        if not prod:
            return "No pude identificar el producto para revisar stock."

        pid = prod["id"]
        pname = prod["nombre"]

        update_state(user_id, {"selected_product_id": pid, "stage": "buy"})
        if (tienda or "").strip():
            update_state(user_id, {"selected_store": tienda.strip()})

        if (tienda or "").strip():
            rows = session.run("""
                MATCH (t:Tienda {nombre:$tienda})-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                RETURN t.nombre AS tienda, s.cantidad AS cantidad
            """, tienda=tienda.strip(), pid=pid)
        else:
            rows = session.run("""
                MATCH (t:Tienda)-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                RETURN t.nombre AS tienda, s.cantidad AS cantidad
                ORDER BY cantidad DESC
            """, pid=pid)

        data = [dict(r) for r in rows]
        if not data:
            return f"❌ No hay stock registrado para {pname}."

        txt = f"✅ Disponibilidad para {pname} [{pid}]:\n"
        for r in data:
            txt += f"- {r['tienda']}: {r['cantidad']} unid.\n"
        return txt


@tool
def verificar_stock_carrito(tienda: str = "", user_id: str = "anonimo") -> str:
    """Revisa stock para TODO el carrito y guarda selected_store (mejor opción) si no se especifica tienda."""
    logger.info(f"🛠️ [TOOL] verificar_stock_carrito | tienda='{tienda}'")
    state = get_state(user_id)
    cart = state.get("cart_items", []) or []
    if not cart:
        return "Tu carrito está vacío. Dime qué productos quieres comprar."

    lines = []
    best_store_to_save = None

    with driver.session() as session:
        for c in cart:
            pid = c.get("id")
            pname = c.get("nombre", pid)
            qty = int(c.get("qty", 1))

            if (tienda or "").strip():
                rows = session.run("""
                    MATCH (t:Tienda {nombre:$tienda})-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                    RETURN t.nombre AS tienda, s.cantidad AS cantidad
                """, tienda=tienda.strip(), pid=pid)
            else:
                rows = session.run("""
                    MATCH (t:Tienda)-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                    RETURN t.nombre AS tienda, s.cantidad AS cantidad
                    ORDER BY cantidad DESC
                """, pid=pid)

            data = [dict(r) for r in rows]
            if not data:
                lines.append(f"❌ {pname} [{pid}] x{qty}: sin stock.")
                continue

            best = data[0]
            ok = int(best["cantidad"]) >= qty
            lines.append(
                f"{'✅' if ok else '⚠️'} {pname} [{pid}] x{qty}: mejor -> {best['tienda']} ({best['cantidad']} unid.)"
            )

            if not (tienda or "").strip() and best_store_to_save is None:
                best_store_to_save = best["tienda"]

    state["stage"] = "buy"
    if best_store_to_save:
        state["selected_store"] = best_store_to_save
    save_state(user_id, state)

    return "Stock del carrito:\n" + "\n".join(lines) + "\n\n" + cart_to_text(state)


@tool
def obtener_contacto_tienda(nombre_tienda: str = "", user_id: str = "anonimo") -> str:
    """Obtiene teléfono/WhatsApp/horario/dirección de una tienda desde Neo4j."""
    state = get_state(user_id)
    if not (nombre_tienda or "").strip():
        if state.get("selected_store"):
            nombre_tienda = state["selected_store"]
        else:
            return "¿De qué tienda? Opciones: Tienda Central, Sucursal Norte, Venta Online."

    logger.info(f"🛠️ [TOOL] obtener_contacto_tienda | tienda='{nombre_tienda}'")

    with driver.session() as session:
        row = session.run("""
            MATCH (t:Tienda)
            WHERE toLower(t.nombre) CONTAINS toLower($name)
            RETURN t.nombre AS nombre, t.canal AS canal, t.telefono AS telefono,
                   t.whatsapp AS whatsapp, t.direccion AS direccion, t.horario AS horario
            LIMIT 1
        """, name=nombre_tienda).single()

    if not row:
        return "No encontré esa tienda. Opciones: Tienda Central, Sucursal Norte, Venta Online."

    update_state(user_id, {"selected_store": row["nombre"], "stage": "contact"})

    def safe(v):
        return v if v not in (None, "") else "N/A"

    return (
        f"📍 {row['nombre']} ({safe(row.get('canal'))})\n"
        f"☎️ Tel: {safe(row.get('telefono'))}\n"
        f"💬 WhatsApp: {safe(row.get('whatsapp'))}\n"
        f"🕒 Horario: {safe(row.get('horario'))}\n"
        f"📌 Dirección: {safe(row.get('direccion'))}\n"
    )


@tool
def registrar_correccion(entidad: str, correccion: str, user_id: str = "anonimo") -> str:
    """Guarda una corrección del usuario en Neo4j (producto/tienda/auto)."""
    logger.info(f"🛠️ [TOOL] registrar_correccion | entidad='{entidad}'")
    state = get_state(user_id)

    texto = (correccion or "").strip()
    if len(texto) < 5:
        return "La corrección es muy corta. Dame un poco más de detalle."

    with driver.session() as session:
        # Tienda
        if entidad == "tienda" or (entidad == "auto" and state.get("selected_store")):
            tienda = state.get("selected_store")
            if not tienda:
                return "¿De qué tienda es la corrección? (Central/Norte/Online)"

            res = session.run("""
                MATCH (t:Tienda {nombre:$tienda})
                CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"manual"})
                MERGE (t)-[:TIENE_CORRECCION]->(c)
                RETURN t.nombre AS entidad
            """, tienda=tienda, texto=texto).single()

            return f"✅ Corrección guardada para tienda: {res['entidad']}" if res else "No pude guardar la corrección de tienda."

        # Producto
        pid = state.get("selected_product_id")
        if not pid:
            v = embedder.encode(texto).tolist()
            res = session.run("""
                CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
                YIELD node AS p, score
                WHERE score > 0.7
                CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"manual"})
                MERGE (p)-[:TIENE_CORRECCION]->(c)
                RETURN p.nombre AS entidad
            """, vector=v, texto=texto).single()
            return f"✅ Corrección guardada para: {res['entidad']}" if res else "Guardé el feedback, pero no pude asociarlo con seguridad."

        res = session.run("""
            MATCH (p:Producto {id:$pid})
            CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"manual"})
            MERGE (p)-[:TIENE_CORRECCION]->(c)
            RETURN p.nombre AS entidad
        """, pid=pid, texto=texto).single()

        return f"✅ Corrección guardada para producto: {res['entidad']}" if res else "No pude guardar la corrección de producto."


@tool
def finalizar_compra(tienda: str = "", user_id: str = "anonimo") -> str:
    """
    Finaliza compra (simulada):
    - Valida carrito
    - Verifica stock del carrito para escoger tienda si no viene 'tienda'
    - Devuelve "✅ Compra realizada" + tienda + contacto + lista de productos
    - Vacía el carrito
    """
    logger.info(f"🛠️ [TOOL] finalizar_compra | tienda='{tienda}'")

    state = get_state(user_id)
    cart = state.get("cart_items", []) or []
    if not cart:
        return "Tu carrito está vacío. Dime qué productos quieres comprar."

    best_store = (tienda or "").strip() or state.get("selected_store")

    with driver.session() as session:
        # Si no hay tienda, elegimos la mejor para el primer item por stock
        if not best_store:
            first = cart[0]
            pid = first.get("id")
            row = session.run("""
                MATCH (t:Tienda)-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                RETURN t.nombre AS tienda, s.cantidad AS cantidad
                ORDER BY cantidad DESC
                LIMIT 1
            """, pid=pid).single()
            if row:
                best_store = row["tienda"]

        if not best_store:
            return "No pude determinar una tienda. ¿Prefieres Tienda Central, Sucursal Norte o Venta Online?"

        # Verificar stock de todo el carrito en esa tienda
        faltantes = []
        for c in cart:
            pid = c.get("id")
            pname = c.get("nombre", pid)
            qty = int(c.get("qty", 1))

            row = session.run("""
                MATCH (t:Tienda {nombre:$tienda})-[s:TIENE_STOCK]->(p:Producto {id:$pid})
                RETURN s.cantidad AS cantidad
                LIMIT 1
            """, tienda=best_store, pid=pid).single()

            if not row or int(row["cantidad"]) < qty:
                faltantes.append(f"- {pname} [{pid}] x{qty}")

        if faltantes:
            return (
                f"⚠️ No hay stock suficiente en **{best_store}** para:\n"
                + "\n".join(faltantes)
                + "\n\nDime si quieres intentar otra tienda (Central/Norte/Online)."
            )

        # Contacto
        t = session.run("""
            MATCH (t:Tienda {nombre:$name})
            RETURN t.nombre AS nombre, t.canal AS canal, t.telefono AS telefono,
                   t.whatsapp AS whatsapp, t.direccion AS direccion, t.horario AS horario
            LIMIT 1
        """, name=best_store).single()

    def safe(v):
        return v if v not in (None, "") else "N/A"

    ticket = cart_to_text(state)

    # Vaciar carrito y set stage
    state["selected_store"] = best_store
    state["stage"] = "done"
    state["cart_items"] = []
    save_state(user_id, state)

    return (
        "✅ **Compra realizada**\n\n"
        f"📍 **Acércate a:** {safe(t['nombre'])} ({safe(t.get('canal'))})\n"
        f"📌 Dirección: {safe(t.get('direccion'))}\n"
        f"🕒 Horario: {safe(t.get('horario'))}\n"
        f"☎️ Tel: {safe(t.get('telefono'))}\n"
        f"💬 WhatsApp: {safe(t.get('whatsapp'))}\n\n"
        "🧾 **Productos comprados:**\n"
        f"{ticket}"
    )


@tool
def buscar_relaciones_grafo(termino: str, user_id: str = "anonimo") -> str:
    """Búsqueda BFS en el grafo (semillas por texto + expansión por relaciones)."""
    logger.info(f"tool_buscar_relaciones_grafo termino='{termino}'")
    
    all_visited_nodes = []
    all_edges = []
    
    def on_search_step(depth: int, node_ids: list, edge_tuples: list):
        """Callback para emitir eventos en cada paso del BFS."""
        all_visited_nodes.extend(node_ids)
        GRAPH_HIGHLIGHTS[user_id] = list(dict.fromkeys(all_visited_nodes))
        edges_data = [{"from": e[0], "to": e[1], "type": e[2]} for e in edge_tuples]
        all_edges.extend(edges_data)
        
        emit_event(user_id, "info", "graph_search_step", {
            "depth": depth,
            "node_ids": node_ids,
            "edges": edges_data,
            "cumulative_nodes": list(dict.fromkeys(all_visited_nodes)),
        })
    
    rows = buscar_en_grafo(
        driver,
        termino=termino,
        limite=5,
        max_depth=2,
        on_step=on_search_step,
        step_delay=0.4,
    )
    
    if rows:
        node_ids = []
        for row in rows:
            seed_id = row.get("seed_id")
            related_id = row.get("related_id")
            if seed_id:
                node_ids.append(seed_id)
            if related_id:
                node_ids.append(related_id)
        unique_nodes = list(dict.fromkeys(node_ids))
        GRAPH_HIGHLIGHTS[user_id] = unique_nodes
        emit_event(user_id, "info", "graph_search_complete", {
            "node_ids": unique_nodes,
            "edges": all_edges,
            "total_depth": 2,
        })
    return format_graph_search_results(rows)


TOOLS = [
    buscar_productos,
    seleccionar_opcion,
    agregar_al_carrito,
    ver_carrito,
    remover_del_carrito,
    vaciar_carrito,
    verificar_stock,
    verificar_stock_carrito,
    obtener_contacto_tienda,
    finalizar_compra,
    registrar_correccion,
    buscar_relaciones_grafo,
]


# ----------------------------------------------------------------------
# 5) PLANNER (FASE 3)
# ----------------------------------------------------------------------
PLANNER_SYSTEM = """
Eres un PLANNER. Devuelve SOLO JSON válido.

Formato:
{"steps":[{"tool":"buscar_productos","args":{"query":"...","user_id":"..."}}]}

REGLAS IMPORTANTES:
- NUNCA inventes productos en el carrito.
- SOLO agrega al carrito si el usuario lo pide explícitamente (ej: "quiero X", "añade X", "me llevo X").

Reglas conversacionales:
- Explorar ("busco", "quiero ver", "recomiéndame") => buscar_productos.

- Elegir opción ("la segunda", "opción 2", "la tercera") => seleccionar_opcion(opcion).

- Agregar al carrito:
  Si el usuario dice "quiero la X", "me llevo X", "añade X", "laptop y mouse" => usar agregar_al_carrito.
  Si menciona DOS cosas ("X y Y"), devuelve DOS steps con agregar_al_carrito (uno por cada item).

- Remover del carrito:
  Si el usuario dice "quita X, Y y Z" => devuelve UN step remover_del_carrito(items_ref="X, Y, Z").
  Si el usuario dice "quita 1 y 3" => devuelve UN step remover_del_carrito(items_ref="1,3").
  SOLO usa vaciar_carrito si el usuario dice explícitamente "vacía el carrito", "borra todo", "elimina todo".

- Después de agregar o remover del carrito, agrega un último step: ver_carrito(user_id).

- Vaciar carrito SOLO si el usuario pide "vacía el carrito", "borra todo", "elimina todo".
  => vaciar_carrito.

- Ver carrito ("qué tengo", "ver carrito", "mi carrito") => ver_carrito.

- Comprar / stock:
  - Si el usuario dice "quiero comprar" y hay carrito => verificar_stock_carrito(tienda opcional).
  - Si el usuario dice "quiero ver stock" (sin carrito) => verificar_stock(producto_ref="", tienda="").

- Finalizar compra ("proceder a compra", "finalizar compra", "comprar ya", "haz la compra", "listo comprar"):
  Si hay carrito => finalizar_compra(tienda="")

- Contacto ("número", "whatsapp", "llamar", "dirección", "horario", "contacto") => obtener_contacto_tienda.

- Corrección ("está mal", "no es así", "el precio real", "corrige") =>
  registrar_correccion(entidad="auto", correccion="...").

- Si el usuario pide buscar relaciones en el grafo, nodos o estructura => buscar_relaciones_grafo(termino, user_id).

- Si no necesitas tools => {"steps":[]}

- Si el usuario pide "dónde comprar" o "muéstrame dónde", responde con:
  (1) tienda/canal recomendado (según stock) y (2) contacto/dirección/horario.
  No vuelvas a preguntar "ver stock o comprar".

- Dónde comprar ("dónde", "muéstrame dónde", "en qué tienda", "a dónde voy", "dónde lo consigo"):
  Si hay carrito:
    1) verificar_stock_carrito(tienda="")
    2) obtener_contacto_tienda(nombre_tienda="")   # usa selected_store del state
  Si NO hay carrito:
    1) verificar_stock(producto_ref="", tienda="")
    2) obtener_contacto_tienda(nombre_tienda="")

IMPORTANTE:
- SIEMPRE incluye "user_id" en args.
- No inventes tools ni args.

Tools válidas:
- buscar_productos(query, user_id)
- seleccionar_opcion(opcion, user_id)
- agregar_al_carrito(producto_ref, qty, user_id)
- remover_del_carrito(items_ref, user_id)
- ver_carrito(user_id)
- vaciar_carrito(user_id)
- verificar_stock(producto_ref, tienda, user_id)
- verificar_stock_carrito(tienda, user_id)
- obtener_contacto_tienda(nombre_tienda, user_id)
- finalizar_compra(tienda, user_id)
- registrar_correccion(entidad, correccion, user_id)
- buscar_relaciones_grafo(termino, user_id)
"""


def planificar(query_text: str, router_label: str, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Genera un plan JSON con herramientas a ejecutar."""
    prompt = (
        f"Usuario: {query_text}\n"
        f"user_id: {user_id}\n"
        f"Estado: {json.dumps(state, ensure_ascii=False)}\n"
        f"Router: {router_label}\n"
        f"Devuelve plan JSON."
    )
    msg = llm.invoke([SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)])
    raw = (msg.content or "").strip()

    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
            raise ValueError("Plan inválido.")
        for s in plan["steps"]:
            if isinstance(s, dict) and isinstance(s.get("args"), dict):
                s["args"]["user_id"] = user_id
        return plan
    except Exception as e:
        logger.warning(f"⚠️ [PLANNER] JSON inválido => steps=[] | err={e} | raw={raw[:200]}")
        return {"steps": []}


def _simulate_tool_execution(tool_name: str, args: Dict[str, Any]) -> str:
    """Simula ejecución de tools con logs tipo print para demos."""
    user_id = args.get("user_id", "usuario_demo")
    steps = [
        f"{tool_name}() - print('iniciando ejecución')",
        f"{tool_name}() - print('args={json.dumps(args, ensure_ascii=False)}')",
        f"{tool_name}() - print('finalizando ejecución')",
    ]
    for line in steps:
        logger.info(line)
    return f"✅ [SIMULADO] {tool_name} completado para {user_id}"


def _invoke_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    if TOOLS_PRINT_ONLY:
        return _simulate_tool_execution(tool_name, args)

    if tool_name == "buscar_productos":
        return buscar_productos.invoke(args)
    if tool_name == "seleccionar_opcion":
        return seleccionar_opcion.invoke(args)
    if tool_name == "agregar_al_carrito":
        return agregar_al_carrito.invoke(args)
    if tool_name == "remover_del_carrito":
        return remover_del_carrito.invoke(args)
    if tool_name == "ver_carrito":
        return ver_carrito.invoke(args)
    if tool_name == "vaciar_carrito":
        return vaciar_carrito.invoke(args)
    if tool_name == "verificar_stock":
        return verificar_stock.invoke(args)
    if tool_name == "verificar_stock_carrito":
        return verificar_stock_carrito.invoke(args)
    if tool_name == "obtener_contacto_tienda":
        return obtener_contacto_tienda.invoke(args)
    if tool_name == "finalizar_compra":
        return finalizar_compra.invoke(args)
    if tool_name == "registrar_correccion":
        return registrar_correccion.invoke(args)
    if tool_name == "buscar_relaciones_grafo":
        return buscar_relaciones_grafo.invoke(args)
    return f"Tool desconocida: {tool_name}"


# ----------------------------------------------------------------------
# 6) EXECUTOR (FASE 4)
# ----------------------------------------------------------------------
def ejecutar_plan(plan: Dict[str, Any], user_id: str, trace_id: str, initial_results: Optional[List[Dict[str, Any]]] = None,
                 start_index: int = 0) -> List[Dict[str, Any]]:
    """Ejecuta cada step del plan y emite eventos para frontend."""
    results = list(initial_results or [])
    steps = plan.get("steps", [])
    emit_event(user_id, "info", "queue_execution_started", {
        "trace_id": trace_id,
        "start_index": start_index,
        "total_steps": len(steps),
        "already_completed": len(results),
    })

    for idx in range(start_index, len(steps)):
        step = steps[idx]
        human_idx = idx + 1
        tool_name = step.get("tool")
        args = step.get("args", {}) or {}
        emit_event(user_id, "info", "queue_step_start", {
            "trace_id": trace_id,
            "step": human_idx,
            "total_steps": len(steps),
            "tool": tool_name,
            "args": args,
            "queue_progress_before": f"{idx}/{len(steps)}",
        })
        logger.info(f"Fase 4: paso {human_idx}/{len(steps)} {tool_name}")

        try:
            out = _invoke_tool(tool_name, args)

            if tool_name != "buscar_relaciones_grafo":
                emit_graph_highlight_for_tool(user_id, tool_name, args)

            results.append({"tool": tool_name, "args": args, "output": out})
            emit_event(user_id, "info", "queue_step_done", {
                "trace_id": trace_id,
                "step": human_idx,
                "total_steps": len(steps),
                "tool": tool_name,
                "output_len": len(str(out)),
                "queue_progress_after": f"{human_idx}/{len(steps)}",
            })
        except Exception as step_error:
            emit_event(user_id, "error", "queue_step_error", {
                "trace_id": trace_id,
                "step": human_idx,
                "total_steps": len(steps),
                "tool": tool_name,
                "error": str(step_error),
            })
            logger.error(f"Fase 4: paso {human_idx} ERROR -> {tool_name}")
            raise

    emit_event(user_id, "info", "queue_execution_finished", {
        "trace_id": trace_id,
        "executed_steps": len(steps) - start_index,
        "total_results": len(results),
    })
    return results


def ejecutar_o_reanudar_cola(plan: Dict[str, Any], user_id: str, trace_id: str) -> List[Dict[str, Any]]:
    pending = get_pending_queue(user_id)
    emit_event(user_id, "info", "queue_orchestrator_enter", {"trace_id": trace_id, "has_pending": bool(pending)})
    if pending and pending.get("steps"):
        emit_event(user_id, "warning", "queue_resume_detected", {
            "previous_trace_id": pending.get("trace_id"),
            "pending_from_step": pending.get("next_index", 0) + 1,
            "pending_total_steps": len(pending.get("steps", [])),
        })
        resumed_results = ejecutar_plan(
            {"steps": pending["steps"]},
            user_id=user_id,
            trace_id=pending.get("trace_id", trace_id),
            initial_results=pending.get("results", []),
            start_index=int(pending.get("next_index", 0)),
        )
        clear_queue(user_id)
        emit_event(user_id, "info", "queue_resume_cleared", {"trace_id": pending.get("trace_id", trace_id)})
        emit_event(user_id, "info", "queue_resume_completed", {
            "resumed_steps": len(resumed_results),
            "trace_id": pending.get("trace_id", trace_id),
        })

    queue_data = {
        "trace_id": trace_id,
        "steps": plan.get("steps", []),
        "next_index": 0,
        "results": [],
    }
    save_queue(user_id, queue_data)
    emit_event(user_id, "info", "queue_snapshot_saved", {"trace_id": trace_id, "next_index": queue_data["next_index"], "results_saved": len(queue_data["results"])})
    emit_event(user_id, "info", "queue_enqueued", {
        "trace_id": trace_id,
        "total_steps": len(queue_data["steps"]),
    })

    results = []
    steps = queue_data["steps"]
    for idx in range(len(steps)):
        queue_data["next_index"] = idx
        save_queue(user_id, queue_data)
        emit_event(user_id, "info", "queue_progress_checkpoint", {"trace_id": trace_id, "next_index": idx, "total_steps": len(steps)})

        partial = ejecutar_plan(
            {"steps": steps[idx:idx+1]},
            user_id=user_id,
            trace_id=trace_id,
            start_index=0,
        )
        if partial:
            queue_data["results"].append(partial[0])
            queue_data["next_index"] = idx + 1
            save_queue(user_id, queue_data)
            emit_event(user_id, "info", "queue_snapshot_saved", {"trace_id": trace_id, "next_index": idx + 1, "results_saved": len(queue_data["results"])})
            results.append(partial[0])

    clear_queue(user_id)
    emit_event(user_id, "info", "queue_cleared", {"trace_id": trace_id})
    emit_event(user_id, "info", "queue_completed", {
        "trace_id": trace_id,
        "total_steps": len(steps),
    })
    return results


# ----------------------------------------------------------------------
# 7) RESPONDER (FASE 5)
# ----------------------------------------------------------------------
RESPONDER_SYSTEM = """
Eres un vendedor conversacional.

Reglas:
- Si el usuario explora, muestra opciones y haz 1 pregunta útil (presupuesto/uso).
- No muestres stock salvo si el usuario lo pidió o dijo que quiere comprar.
- Si el usuario agrega al carrito, confirma el carrito y pregunta: "¿quieres ver stock o comprar?"
- Si el usuario pide quitar un ítem, confirma el cambio y muestra el carrito actualizado.
- Si el usuario dice "quiero comprar" y hay carrito, sugiere ver stock del carrito.
- Si se ejecutó finalizar_compra, NO preguntes "ver stock o comprar". Solo confirma compra y da tienda/productos.
- Si entregas contacto, hazlo claro.
- Si no hay herramientas, haz una pregunta corta para aclarar.
- Si aparecen correcciones aprendidas, trátalas como verdad.
"""


def redactar_respuesta(query_text: str, tool_results: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
    """Redacta respuesta final usando el contexto de herramientas."""
    contexto = ""
    for r in tool_results:
        contexto += f"[TOOL={r['tool']} ARGS={r['args']}]\n{r['output']}\n\n"

    msg = llm.invoke([
        SystemMessage(content=RESPONDER_SYSTEM),
        HumanMessage(content=(
            f"Usuario:\n{query_text}\n\n"
            f"Estado actual:\n{json.dumps(state, ensure_ascii=False)}\n\n"
            f"Contexto tools:\n{contexto}"
        ))
    ])
    return msg.content


# ----------------------------------------------------------------------
# 8) PIPELINE 1→5
# ----------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    query_text: str
    trace_id: str
    user_id: str
    state: Dict[str, Any]
    q_vec: List[float]
    router_label: str
    router_score: float
    sims: List[float]
    router_strategy: str
    plan: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    state_after: Dict[str, Any]
    response: str


def node_embedding(state: PipelineState) -> PipelineState:
    query_text = state["query_text"]
    logger.info(f"Fase 1: query = \"{query_text}\"")
    q_vec = embedder.encode(query_text, normalize_embeddings=True).tolist()
    logger.info("Fase 2: embedding")
    return {"q_vec": q_vec, "state": get_state(state["user_id"])}


def node_function_selection(state: PipelineState) -> PipelineState:
    router_label, router_score, sims, router_strategy = seleccionar_funcion(state["query_text"], state["q_vec"])
    ranked = sorted(
        [{"label": ROUTER_LABELS[i], "score": float(sims[i])} for i in range(len(ROUTER_LABELS))],
        key=lambda x: x["score"],
        reverse=True,
    )
    emit_event(
        state["user_id"],
        "info",
        "function_selected",
        {
            "trace_id": state["trace_id"],
            "selected_function": router_label,
            "selected_score": round(float(router_score), 4),
            "selection_strategy": router_strategy,
            "top_candidates": ranked[:3],
        },
    )
    logger.info(f"Fase 3: Function selection = \"{router_label}\" (score={round(float(router_score), 4)})")
    update_state(state["user_id"], {"last_intent": router_label})
    return {
        "router_label": router_label,
        "router_score": router_score,
        "sims": sims,
        "router_strategy": router_strategy,
    }


def node_plan(state: PipelineState) -> PipelineState:
    logger.info("Fase 4: planificacion")
    plan = planificar(state["query_text"], state["router_label"], state["user_id"], state["state"])
    plan_tools = [step.get("tool") for step in plan.get("steps", []) if isinstance(step, dict)]
    logger.info(f"Fase 4: plan = {plan_tools}")
    return {"plan": plan}


def node_execute(state: PipelineState) -> PipelineState:
    emit_event(state["user_id"], "info", "phase_4_start", {"trace_id": state["trace_id"]})
    tool_results = ejecutar_o_reanudar_cola(state["plan"], user_id=state["user_id"], trace_id=state["trace_id"])
    emit_event(state["user_id"], "info", "phase_4_done", {"trace_id": state["trace_id"], "tools": len(tool_results)})
    return {"tool_results": tool_results, "state_after": get_state(state["user_id"])}


def node_response(state: PipelineState) -> PipelineState:
    logger.info("Fase 5: respuesta")
    response = redactar_respuesta(state["query_text"], state["tool_results"], state["state_after"])
    logger.info("Fase 5: respuesta OK")
    return {"response": response}


def build_orchestration_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("embedding", node_embedding)
    builder.add_node("function_selection", node_function_selection)
    builder.add_node("plan", node_plan)
    builder.add_node("execute", node_execute)
    builder.add_node("response", node_response)

    builder.set_entry_point("embedding")
    builder.add_edge("embedding", "function_selection")
    builder.add_edge("function_selection", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "response")
    builder.add_edge("response", END)
    return builder.compile()


def ejecutar_pipeline(query_text: str, trace_id: str, user_id: str) -> Dict[str, Any]:
    """Ejecuta pipeline orquestado con LangGraph: embedding -> router -> planner -> tools -> responder."""
    global ORCHESTRATION_GRAPH
    if ORCHESTRATION_GRAPH is None:
        ORCHESTRATION_GRAPH = build_orchestration_graph()

    result = ORCHESTRATION_GRAPH.invoke({
        "query_text": query_text,
        "trace_id": trace_id,
        "user_id": user_id,
    })

    return {
        "response": result["response"],
        "router": {
            "label": result["router_label"],
            "score": result["router_score"],
            "sims": result["sims"],
            "strategy": result["router_strategy"],
        },
        "plan": result["plan"],
        "tool_results": result["tool_results"],
        "state": result["state_after"],
    }


# ----------------------------------------------------------------------
# 9) API
# ----------------------------------------------------------------------
def _display_label(node_id: int, labels: List[str], props: Dict[str, Any]) -> str:
    for key in ("nombre", "name", "titulo", "id"):
        val = props.get(key)
        if val:
            return str(val)
    if labels:
        return labels[0]
    return str(node_id)


def fetch_graph_snapshot(limit_nodes: int = 50, limit_edges: int = 80, highlight_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    if not driver:
        return {"nodes": [], "edges": []}

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    edge_keys = set()

    highlight_ids = highlight_ids or []

    cypher_highlight = """
    MATCH (n)
    WHERE elementId(n) IN $highlight_ids
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN elementId(n) AS n_id, labels(n) AS n_labels, properties(n) AS n_props,
           elementId(m) AS m_id, labels(m) AS m_labels, properties(m) AS m_props,
           type(r) AS r_type
    """

    cypher_sample = """
    MATCH (n)
    WITH n LIMIT $limit_nodes
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN elementId(n) AS n_id, labels(n) AS n_labels, properties(n) AS n_props,
           elementId(m) AS m_id, labels(m) AS m_labels, properties(m) AS m_props,
           type(r) AS r_type
    LIMIT $limit_edges
    """

    def add_row(row: Any) -> None:
        n_id = row.get("n_id")
        n_labels = row.get("n_labels") or []
        n_props = row.get("n_props") or {}
        if n_id is not None and n_id not in nodes:
            nodes[n_id] = {
                "id": n_id,
                "label": _display_label(n_id, n_labels, n_props),
                "labels": n_labels,
                "props": n_props,
            }

        m_id = row.get("m_id")
        m_labels = row.get("m_labels") or []
        m_props = row.get("m_props") or {}
        if m_id is not None and m_id not in nodes:
            nodes[m_id] = {
                "id": m_id,
                "label": _display_label(m_id, m_labels, m_props),
                "labels": m_labels,
                "props": m_props,
            }

        rel_type = row.get("r_type")
        if rel_type and n_id is not None and m_id is not None:
            key = f"{n_id}|{m_id}|{rel_type}"
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append({
                    "from": n_id,
                    "to": m_id,
                    "type": rel_type,
                })

    with driver.session() as session:
        if highlight_ids:
            highlight_rows = session.run(cypher_highlight, highlight_ids=highlight_ids)
            for row in highlight_rows:
                add_row(row)

        sample_rows = session.run(cypher_sample, limit_nodes=limit_nodes, limit_edges=limit_edges)
        for row in sample_rows:
            add_row(row)

    return {"nodes": list(nodes.values()), "edges": edges}
class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonimo"


class FeedbackRequest(BaseModel):
    run_id: str
    score: int
    comment: Optional[str] = None
    user_id: str = "anonimo"


@app.get("/graph")
async def graph_endpoint(
    limit_nodes: int = Query(default=50, ge=1, le=200),
    limit_edges: int = Query(default=80, ge=0, le=300),
    user_id: Optional[str] = None,
):
    highlight_ids = GRAPH_HIGHLIGHTS.get(user_id or "", [])
    return fetch_graph_snapshot(limit_nodes=limit_nodes, limit_edges=limit_edges, highlight_ids=highlight_ids)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    trace_id = str(uuid.uuid4())[:8]
    logger.info("chat_request")

    if is_off_topic_query(req.query):
        logger.info("chat_off_topic_filtered")
        return {"response": off_topic_response(), "run_id": None, "trace_id": trace_id, "status": "filtered"}

    start = perf_counter()
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            result = ejecutar_pipeline(req.query, trace_id, req.user_id)
            latency_ms = (perf_counter() - start) * 1000

            mlflow.log_param("user_id", req.user_id)
            mlflow.log_param("trace_id", trace_id)
            mlflow.log_param("router_label", result["router"]["label"])
            mlflow.log_metric("router_score", float(result["router"]["score"]))
            mlflow.log_metric("latency_ms", float(latency_ms))
            mlflow.log_text(json.dumps(result["plan"], ensure_ascii=False, indent=2), "plan.json")
            mlflow.log_text(result["response"], "respuesta_agente.txt")
            mlflow.log_text(json.dumps(result["state"], ensure_ascii=False, indent=2), "state.json")

            logger.info("chat_success")
            return {"response": result["response"], "run_id": run_id, "trace_id": trace_id, "status": "success"}

    except Exception as e:
        logger.error("chat_error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/{user_id}")
async def events_endpoint(user_id: str, limit: int = Query(default=50, ge=1, le=500)):
    if not rdb:
        return {"events": []}
    key = _events_key(user_id)
    raw_items = rdb.lrange(key, 0, limit - 1)
    if raw_items:
        rdb.ltrim(key, len(raw_items), -1)
    events = []
    for item in raw_items:
        try:
            events.append(json.loads(item))
        except Exception:
            events.append({"level": "info", "event": "unparsed", "payload": {"raw": item}})
    return {"events": events}

@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    logger.info(f"⭐ [HTTP] /feedback | run_id={req.run_id} | score={req.score}")

    try:
        with mlflow.start_run(run_id=req.run_id):
            mlflow.log_metric("helpfulness", req.score)
            if req.comment:
                mlflow.log_text(req.comment, "feedback_usuario.txt")

        # Auto-learn: usa focus actual (selected_product_id / selected_store)
        if req.score == 0 and req.comment and len(req.comment.strip()) > 5:
            texto = req.comment.strip()
            state = get_state(req.user_id)
            pid = state.get("selected_product_id")
            store = state.get("selected_store")

            logger.info(f"🧠 [AUTO-LEARN] pid={pid} store={store} texto='{texto}'")

            with driver.session() as session:
                if pid:
                    res = session.run("""
                        MATCH (p:Producto {id:$pid})
                        CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"feedback_loop"})
                        MERGE (p)-[:TIENE_CORRECCION]->(c)
                        RETURN p.nombre AS entidad
                    """, pid=pid, texto=texto).single()
                    if res:
                        return {"message": f"Gracias. Aprendí una corrección sobre: {res['entidad']}."}

                if store:
                    res = session.run("""
                        MATCH (t:Tienda {nombre:$store})
                        CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"feedback_loop"})
                        MERGE (t)-[:TIENE_CORRECCION]->(c)
                        RETURN t.nombre AS entidad
                    """, store=store, texto=texto).single()
                    if res:
                        return {"message": f"Gracias. Aprendí una corrección sobre la tienda: {res['entidad']}."}

                v = embedder.encode(texto).tolist()
                res = session.run("""
                    CALL db.index.vector.queryNodes('productos_embeddings', 1, $vector)
                    YIELD node AS p, score
                    WHERE score > 0.7
                    CREATE (c:Aprendizaje {nota:"CORRECCIÓN DE USUARIO: " + $texto, fecha: datetime(), origen:"feedback_loop"})
                    MERGE (p)-[:TIENE_CORRECCION]->(c)
                    RETURN p.nombre AS entidad
                """, vector=v, texto=texto).single()

                if res:
                    return {"message": f"Gracias. Aprendí una corrección sobre: {res['entidad']}."}

            return {"message": "Feedback recibido (no pude asociarlo con seguridad)"}

        return {"message": "Feedback recibido"}

    except Exception as e:
        logger.error(f"❌ /feedback error: {e}")
        return {"message": "Error procesando feedback"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)