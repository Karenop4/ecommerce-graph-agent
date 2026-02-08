# Grafo-Conocimiento: AI Agent for E-Commerce Knowledge Graph

A sophisticated e-commerce intelligent agent powered by AI that leverages a knowledge graph (Neo4j) for semantic product search, inventory management, and continuous learning through MLOps.

## Tech Stack

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E4?style=for-the-badge&logo=mlflow&logoColor=white)

### Frontend
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)

### Infrastructure
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker%20Compose](https://img.shields.io/badge/Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Notebooks](#notebooks)

---

##  Project Overview

**Grafo-Conocimiento** is an advanced e-commerce system that implements an AI-powered agent capable of:

- **Semantic Product Search**: Uses embeddings to understand user intent and find relevant products
- **Knowledge Graph Management**: Stores products, accessories, relationships, and inventory data in Neo4j
- **Inventory Tracking**: Monitors product stock across multiple stores in real-time
- **Learning & Feedback Loop**: Records user corrections to improve future recommendations
- **MLOps Integration**: Tracks model performance and experiment iterations through MLflow
- **RESTful API**: FastAPI backend for seamless integration
- **Modern Frontend**: React-based interactive user interface

---

##  Architecture

The system follows a microservices architecture with Docker containerization:

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                        │
│            Interactive User Interface (Port 5173)           │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/axios
┌──────────────────────▼──────────────────────────────────────┐
│                   BACKEND (FastAPI)                         │
│         AI Agent API Server (Port 8000)                     │
│  - LangChain Orchestration                                  │
│  - OpenAI/GPT-4o-mini LLM                                   │
│  - Tool-based agent with Neo4j queries                      │
│  - MLflow Tracking                                          │
└──────────┬───────────────┬──────────────────────┬───────────┘
           │               │                      │
           │ (Bolt)        │ (HTTP)               │ (HTTP)
           ▼               ▼                      ▼
   ┌─────────────────┐  ┌──────────────┐   ┌──────────────┐
   │ NEO4J DATABASE  │  │   MLFLOW     │   │  OPENAI API  │
   │  (Port 7687)    │  │ (Port 5000)  │   │  (Cloud)     │
   │                 │  │              │   │              │
   │ - Products      │  │ - Experiments│   │ - GPT-4o-mini│
   │ - Inventory     │  │ - Metrics    │   │ - Embeddings │
   │ - Relationships │  │ - Artifacts  │   │              │
   └─────────────────┘  └──────────────┘   └──────────────┘
```

---

## Detailed Technology Stack

### Backend Stack
| Technology | Purpose | Version |
|-----------|---------|---------|
| **FastAPI** | Python web framework | Latest |
| **LangChain** | LLM orchestration & agent framework | Latest |
| **OpenAI API** | Language model (GPT-4o-mini) | Latest |
| **Neo4j** | Graph database with vector indexing | 5.15.0 |
| **Sentence-Transformers** | Embeddings (all-MiniLM-L6-v2) | Latest |
| **Uvicorn** | ASGI server | Latest |
| **Python-dotenv** | Environment variable management | Latest |
| **Pydantic** | Data validation | Latest |
| **MLflow** | Experiment tracking & model registry | 2.10.0 |

### Frontend Stack
| Technology | Purpose | Version |
|-----------|---------|---------|
| **React** | JavaScript UI library | 19.2.0 |
| **Vite** | Lightning-fast module bundler | 7.2.4 |
| **Axios** | Promise-based HTTP requests | ^1.13.4 |
| **React-Markdown** | Markdown rendering | ^10.1.0 |
| **Lucide-React** | Modern SVG icons | ^0.563.0 |
| **ESLint** | Code quality & linting | ^9.39.1 |

### Infrastructure
| Technology | Purpose | Version |
|-----------|---------|---------|
| **Docker** | Container platform | Latest |
| **Docker Compose** | Multi-container orchestration | Latest |
| **Neo4j** | Graph database | 5.15.0 |
| **MLflow** | MLOps platform | v2.10.0 |

### Development Environment
- **Python**: 3.8+ 
- **Node.js**: LTS (16+)
- **Operating System**: Windows / Linux / MacOS (cross-platform)

---

## Project Structure

```
grafo-conocimiento/
│
├── README.md                           # This file
├── compose.yml                         # Docker Compose configuration
│
├── app/                                # Backend (FastAPI)
│   ├── main.py                         # Main API server & agent logic
│   ├── requirements.txt                # Python dependencies
│   └── Dockerfile                      # Backend container definition
│
├── frontend-agente/                    # Frontend (React + Vite)
│   ├── package.json                    # Node.js dependencies
│   ├── vite.config.js                  # Vite configuration
│   ├── eslint.config.js               # ESLint rules
│   ├── index.html                      # HTML entry point
│   ├── src/                            # React source code
│   │   ├── main.jsx                    # React entry point
│   │   ├── App.jsx                     # Main component
│   │   ├── App.css                     # Styling
│   │   ├── index.css                   # Global styles
│   │   └── assets/                     # Static assets
│   ├── public/                         # Public files
│   └── Dockerfile                      # Frontend container definition
│
├── Notebooks/                          # Jupyter Notebooks
│   ├── 1_etl_graph_population.ipynb   # Data loading & graph setup
│   └── 2_agent_mlflow_experiment.ipynb # Agent experimentation & tracking
│
├── neo4j_data/                         # Neo4j database volume
│   ├── databases/                      # Database files
│   ├── dbms/                           # DBMS configuration
│   └── transactions/                   # Transaction logs
│
├── mlflow_artifacts/                   # MLflow experiment artifacts
├── mlflow_db/                          # MLflow database

```

---

## Functions, Queries, and Embeddings (Neo4j + Redis + OpenAI)

| Function / Tool | Purpose | DB / Service | Read/Write State | Query (Cypher / Redis) | Embeddings (model + where) | Input examples | Output example |
|---|---|---|---|---|---|---|---|
| `buscar_productos(query, user_id)` | Finds similar products by text (catalog, NOT stock). Returns a list and stores candidates for ordinals. | **Neo4j** (Vector Index) + Redis (state) | **Write:** `last_candidates`, `selected_product_id`, `stage=decide` | **Cypher (Vector Search):**<br>`CALL db.index.vector.queryNodes('productos_embeddings', 5, $vector) YIELD node AS p, score WHERE score > 0.5 ... RETURN p.id, p.nombre, p.precio, p.descripcion, collect(acc), collect(correcciones)` | **SentenceTransformers**: `all-MiniLM-L6-v2`<br>Used for `vector = embed(query)`<br>**Index:** `productos_embeddings` (Neo4j) | `"lightweight laptop for travel"` | List like:<br>`1) [L1] MacBook Air M2 ($1200)` |
| `seleccionar_opcion(opcion, user_id)` | Selects the product "second/2/third" based on `last_candidates`. | Redis | **Read:** `last_candidates`<br>**Write:** `selected_product_id`, `stage=decide` | **Redis key:** `session:{user_id}` (JSON) | No embeddings | `"the second"` / `"2"` | `"Selected: Dell XPS 13 [L2]"` |
| `agregar_al_carrito(producto_ref, qty, user_id)` | Adds a product to cart by ordinal, id, or approximate name. Increments qty if it already exists. | Neo4j + Redis | **Read:** `last_candidates` (if ordinal), `cart_items`<br>**Write:** `cart_items`, `selected_product_id`, `stage=decide` | **(a) Ordinal:** uses `last_candidates` (Redis)<br>**(b) ID:** `MATCH (p:Producto {id:$id}) RETURN ...`<br>**(c) Name:** vector search top-1 with score>0.6 | **all-MiniLM-L6-v2** if name-based (not id)<br>Query `productos_embeddings` (Neo4j) | `"the first"` / `"L1"` / `"MacBook Air"` | `"✅ Added... + updated cart"` |
| `remover_del_carrito(items_ref, user_id)` | Removes one or more items from cart by name/id or by "item 1, 3". Does NOT empty all. | Redis | **Read/Write:** `cart_items` | **Redis key:** `session:{user_id}` (JSON)<br>Parsing: split by `,` and `" and "` or indices | No embeddings | `"remove logitech, razer and dell"` / `"remove 1 and 3"` | `"✅ Removed... + updated cart"` |
| `ver_carrito(user_id)` | Shows cart and total. | Redis | **Read:** `cart_items` | **Redis key:** `session:{user_id}` | No embeddings | `"view cart"` | Text with total and subtotals |
| `vaciar_carrito(user_id)` | Empties the cart ONLY when explicitly requested. | Redis | **Write:** `cart_items=[]` | **Redis key:** `session:{user_id}` | No embeddings | `"empty the cart"` | `"🧹 Done. Cart emptied."` |
| `verificar_stock(producto_ref, tienda, user_id)` | Checks stock per store for a product. If `producto_ref` is empty, uses `selected_product_id`. | Neo4j + Redis | **Read:** `selected_product_id` (if `producto_ref` empty)<br>**Write:** `selected_product_id`, `selected_store` (if store), `stage=buy` | **Resolve product:**<br>- ID: `MATCH (p:Producto {id:$id}) ...`<br>- Name: vector search score>0.6<br>**Stock:** `MATCH (t:Tienda)-[s:TIENE_STOCK]->(p:Producto {id:$pid}) RETURN t.nombre, s.cantidad ORDER BY cantidad DESC` | **all-MiniLM-L6-v2** only if resolved by name | `"check stock for MacBook Air"` / `"stock L1"` / `"stock at Central Store"` | Stock list per store |
| `verificar_stock_carrito(tienda, user_id)` | Checks stock for the entire cart. If no store is provided, saves `selected_store` as the best store for the first item. | Neo4j + Redis | **Read:** `cart_items`<br>**Write:** `selected_store` (auto), `stage=buy` | For each item:<br>`MATCH (t:Tienda)-[s:TIENE_STOCK]->(p:Producto {id:$pid}) RETURN t.nombre, s.cantidad ORDER BY cantidad DESC` | No embeddings (ids already in cart) | `"I want to buy"` / `"check cart stock"` | Summary per product + recommended store |
| `obtener_contacto_tienda(nombre_tienda, user_id)` | Returns phone/WhatsApp/hours/address. If name is empty, uses `selected_store`. | Neo4j + Redis | **Read:** `selected_store` (fallback)<br>**Write:** `selected_store`, `stage=contact` | `MATCH (t:Tienda) WHERE toLower(t.nombre) CONTAINS toLower($name) RETURN ... LIMIT 1` | No embeddings | `"give me the WhatsApp"` / `"contact Central Store"` | Contact card (text) |
| `finalizar_compra(tienda, user_id)` *(recommended)* | "Proceed to purchase" flow: validates stock at a store, returns "✅ Purchase completed", shows store/contact and products, and empties cart. | Neo4j + Redis | **Read:** `cart_items`, `selected_store`<br>**Write:** `stage=done`, `cart_items=[]` | 1) Choose store (if missing): top-1 stock for first item<br>2) Validate stock per item at that store<br>3) Fetch store contact | No embeddings | `"proceed to purchase"` / `"finish purchase"` | `"✅ Purchase completed... Visit... Products purchased..."` |
| `registrar_correccion(entidad, correccion, user_id)` | Saves user corrections (product/store) as `Aprendizaje` nodes. | Neo4j + Redis | **Read:** `selected_product_id` or `selected_store` for auto-association | **Product:** `MATCH (p:Producto {id:$pid}) CREATE (c:Aprendizaje {...}) MERGE (p)-[:TIENE_CORRECCION]->(c)`<br>**Store:** `MATCH (t:Tienda {nombre:$tienda}) CREATE ... MERGE (t)-[:TIENE_CORRECCION]->(c)`<br>**Embedding fallback:** vector search score>0.7 | **all-MiniLM-L6-v2** only in fallback (to associate a product) | `"correction: that price is wrong"` | `"✅ Correction saved for..."` |

---

## Database and Structure (Summary)
- **Neo4j**:
   - `(:Producto {id, nombre, precio, descripcion})`
   - `(:Tienda {nombre, canal, telefono, whatsapp, direccion, horario})`
   - `(:Aprendizaje {nota, fecha, origen})`
   - Relationships:
      - `(t:Tienda)-[:TIENE_STOCK {cantidad}]->(p:Producto)`
      - `(p:Producto)-[:COMPATIBLE_CON]->(acc:Producto)`
      - `(p:Producto)-[:TIENE_CORRECCION]->(c:Aprendizaje)` / `(t:Tienda)-[:TIENE_CORRECCION]->(c:Aprendizaje)`
   - **Vector index**: `productos_embeddings` over product embeddings.

- **Redis**:
   - Key: `session:{user_id}`
   - JSON value:
      - `stage`, `selected_product_id`, `selected_store`, `last_candidates`, `cart_items`, `last_intent`

- **Embeddings**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Usage:
      - User query → vector → `productos_embeddings`
      - (optional) associate corrections by similarity when there is no `selected_product_id`

---

## Features

### 1. **Intelligent Product Search**
- Uses vector embeddings to understand semantic meaning
- Finds products even with non-exact queries
- Returns product details: name, price, description
- Suggests compatible accessories automatically

### 2. **Real-time Inventory Management**
- Tracks stock levels across multiple store locations
- Updates availability instantly
- Identifies stockouts and low inventory

### 3. **Knowledge Graph**
- Products stored as nodes with properties (name, price, description)
- Relationships represent compatibility and accessories
- Corrections/learnings stored as feedback nodes
- Vector index for fast semantic similarity search

### 4. **AI Agent Capabilities**
- **Natural Language Understanding**: Interprets user intent
- **Tool-based Actions**: Uses Neo4j queries as tools
- **Context Awareness**: Maintains conversation state
- **Error Recovery**: Handles failures gracefully

### 5. **MLOps & Experimentation**
- Tracks model performance metrics
- Logs experiment runs with parameters
- Compares multiple agent iterations
- Artifacts storage for reproducibility

### 6. **Responsive Frontend**
- Clean, intuitive user interface
- Real-time message streaming
- Hot module reloading during development
- Accessible design with proper UI components

---

## Getting Started

### Prerequisites
- Docker & Docker Compose (recommended)
- Python 3.8+ (if running locally without Docker)
- Node.js 16+ (if running frontend locally)
- OpenAI API key
- Neo4j credentials

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_TRACKING_URI_LOCAL=http://localhost:5000

# Frontend
VITE_API_URL=http://localhost:8000
```

---

## Setup Instructions

### Option 1: Using Docker Compose (Recommended)

**1. Clone or navigate to the project:**
```bash
cd grafo-conocimiento
```

**2. Create `.env` file** with all required environment variables (see above)

**3. Start all services:**
```bash
docker-compose up -d
```

This starts:
- Neo4j database (port 7474 for HTTP, 7687 for Bolt)
- FastAPI backend (port 8000)
- React frontend (port 5173)
- MLflow server (port 5000)

**4. Populate the database** using the Jupyter notebook:
```bash
# Access the notebooks to initialize the graph
jupyter notebook Notebooks/
# Run: 1_etl_graph_population.ipynb
```

### Option 2: Local Development Setup

**Backend:**
```bash
cd app
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend-agente
npm install
npm run dev
```

**Neo4j:**
- Run separately or use Docker: `docker run -d -p 7474:7474 -p 7687:7687 neo4j:5.15.0`

---

## Running the Project

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f backend    # Backend logs
docker-compose logs -f frontend   # Frontend logs
docker-compose logs -f neo4j      # Database logs
```

### Access Applications
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7474
- **MLflow UI**: http://localhost:5000

### Stop Services
```bash
docker-compose down
```

### Clean Everything (Careful!)
```bash
docker-compose down -v  # Removes volumes too
```

---

## Usage

### Through the Frontend UI
1. Navigate to http://localhost:5173
2. Type a natural language query (e.g., "I need a gaming laptop with wireless mouse")
3. The agent processes your request and returns results
4. View product details, prices, and store availability
5. Provide feedback if recommendations need adjustment

### Through the Backend API
```bash
# Example query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me available MacBooks", "user_id": "user123"}'
```

### Query Examples
- "I need a laptop for programming"
- "Show gaming laptops under $1500"
- "What accessories work with MacBook Air?"
- "Check stock availability in the Central store"
- "Recommend a budget-friendly monitor"

---

## Notebooks

### 1. ETL & Graph Population (`1_etl_graph_population.ipynb`)
**Purpose**: Initialize the knowledge graph with sample data

**Workflow**:
- Connects to Neo4j
- Loads product catalog (laptops, accessories, monitors, etc.)
- Creates semantic embeddings for each product
- Establishes product relationships (compatibility, accessories)
- Initializes store inventory
- Sets up vector index for semantic search

**Key Operations**:
```python
- Creates nodes: :Producto, :Tienda, :Aprendizaje
- Creates relationships: :COMPATIBLE_CON, :TIENE_STOCK, :TIENE_CORRECCION
- Populates vector index on product descriptions
```

**Output**: Fully initialized Neo4j graph with sample e-commerce data

### 2. Agent & MLflow Experiments (`2_agent_mlflow_experiment.ipynb`)
**Purpose**: Test and track AI agent performance

**Workflow**:
- Loads embeddings and LLM
- Defines agent tools (product search, inventory check)
- Runs test queries
- Tracks metrics in MLflow
- Compares experiment iterations
- Records feedback and corrections

**Key Operations**:
```python
- Tests semantic search accuracy
- Evaluates LLM response quality
- Logs execution metrics (latency, token usage)
- Tracks A/B comparisons
- Records user feedback corrections
```

**Output**: MLflow experiments with performance metrics and artifacts

---

## Data Flow

### User Query Journey
```
1. User types query in Frontend (React)
   ↓
2. Frontend sends HTTP request to Backend (FastAPI)
   ↓
3. Backend receives query and LLM creates a plan
   ↓
4. LLM decides which tools to call:
   - consultar_producto() → Searches Neo4j with embeddings
   - check_inventory() → Gets stock from stores
   ↓
5. Tools execute Cypher queries against Neo4j
   ↓
6. LLM processes results and crafts response
   ↓
7. MLflow logs metrics and results
   ↓
8. Backend returns formatted response to Frontend
   ↓
9. Frontend displays results to user
   ↓
10. User provides feedback (optional)
    ↓
11. Feedback stored in Neo4j as learning data
```

---

## Security Notes

- **API Keys**: Never commit `.env` files. Use environment variables.
- **Database**: Change default Neo4j credentials in production.
- **CORS**: Currently allows all origins (`allow_origins=["*"]`). Restrict in production.
- **OpenAI**: Secure your API key. Monitor usage to avoid unexpected charges.

---

## Monitoring & Debugging

### MLflow Experiments
- View at: http://localhost:5000
- Track model performance over time
- Compare different agent configurations
- Export metrics and artifacts

### Neo4j Browser
- Access at: http://localhost:7474
- Visual graph exploration
- Query execution with Cypher
- Performance monitoring

### Backend Logs
```bash
docker-compose logs backend --tail 100
```

### Frontend Console
- Open DevTools (F12) in browser
- Check Network tab for API calls
- View JavaScript console for errors

---

## Troubleshooting

### Neo4j Connection Issues
- Ensure `NEO4J_URI` in `.env` matches service name or localhost
- For local development: change `bolt://neo4j:7687` to `bolt://localhost:7687`
- Check credentials match environment variables

### Frontend API Calls Failing
- Verify `VITE_API_URL` points to correct backend
- Check CORS settings in `main.py`
- Ensure backend is running: `docker-compose logs backend`

### OpenAI API Errors
- Validate API key in `.env`
- Check token usage limits
- Ensure account has sufficient credits

### Docker Build Issues
- Clear cache: `docker-compose build --no-cache`
- Remove stopped containers: `docker system prune`
- Check Docker daemon is running

---

## Development Workflow

### Code Changes
**Backend**: Changes auto-reload due to volume mount
```bash
# Edit app/main.py → changes reflected instantly
```

**Frontend**: Hot module reloading enabled
```bash
# Edit frontend-agente/src/ → browser auto-refreshes
```

### Running Notebooks Locally
```bash
pip install jupyter notebook
jupyter notebook Notebooks/
# Run cells to populate/experiment with the graph
```

### Adding New Tools to Agent
1. Define new tool function in `app/main.py`
2. Decorate with `@tool`
3. Add to agent's toolkit
4. Test in notebook
5. Deploy via Docker

---

## Performance Optimization

### Vector Search
- Embeddings cached in memory after first load
- Neo4j vector index optimized for `all-MiniLM-L6-v2`
- Query similarity threshold: 0.6

### MLOps
- Async MLflow logging to avoid blocking
- Artifact batching for efficient storage
- Experiment metadata indexed

### Frontend
- Vite bundle optimization
- React component lazy loading
- HTTP request caching with Axios

---

## Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [Neo4j Graph Database](https://neo4j.com/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [React 19 Documentation](https://react.dev/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Sentence Transformers](https://www.sbert.net/)

---

## License

This project is part of the **Stochastic Models** course - 6th Cycle.

---

## Author

**Karen Ortiz** - Computer Science Engineering

---

## Acknowledgments

- OpenAI for GPT-4o-mini LLM
- Neo4j for graph database technology
- LangChain for agent orchestration framework
- MLflow for MLOps capabilities
- React & Vite communities for modern web tooling

---

**Last Updated**: February 1, 2026

---

### Quick Commands Reference

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f backend

# Access services
curl http://localhost:8000/docs           # API docs (Swagger)
http://localhost:5173                      # Frontend
http://localhost:7474                      # Neo4j Browser

# Develop
docker-compose down
cd app && python main.py                   # Local backend
cd frontend-agente && npm run dev          # Local frontend

# Clean
docker-compose down -v                     # Remove volumes
docker system prune -a                     # Remove all images
```
