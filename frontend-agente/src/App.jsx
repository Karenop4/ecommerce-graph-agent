import { useState, useRef, useEffect, useMemo } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { ArrowUp, User, Bot, ThumbsUp, ThumbsDown, Send } from "lucide-react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const resolveUserId = () => {
  const key = "ecommerce_agent_user_id";
  const existing = window.sessionStorage.getItem(key);
  if (existing) return existing;
  const generated = `usuario_${Math.random().toString(16).slice(2, 10)}`;
  window.sessionStorage.setItem(key, generated);
  return generated;
};

const USER_ID = resolveUserId();

const GRAPH_SEARCH_TOOLS = new Set([
  "buscar_productos",
  "buscar_relaciones_grafo",
  "verificar_stock",
  "verificar_stock_carrito",
  "obtener_contacto_tienda",
]);

const GRAPH_CANVAS = { width: 320, height: 220 };

const normalizeGraphId = (value) => String(value ?? "");

function App() {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      content:
        "¡Hola! Soy tu asistente de ventas experto. 🤖\nPuedo ayudarte a encontrar productos o recibir tus correcciones si me equivoco.",
      type: "text",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [graphSteps, setGraphSteps] = useState([]);
  const [graphPulses, setGraphPulses] = useState([]);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState([]);
  const [highlightedEdges, setHighlightedEdges] = useState([]);
  const [searchDepth, setSearchDepth] = useState(-1);
  const [graphMeta, setGraphMeta] = useState({
    status: "idle",
    activeTool: "",
    traceId: "",
    lastEvent: 0,
    currentDepth: -1,
  });
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });

  const [activeFeedback, setActiveFeedback] = useState(null); // { runId, index }
  const [commentText, setCommentText] = useState("");

  const messagesEndRef = useRef(null);
  const feedbackTextareaRef = useRef(null);
  const pulseTimersRef = useRef([]);
  const highlightTimerRef = useRef(null);
  const flowTimersRef = useRef([]);
  const lastEventsErrorRef = useRef(0);
  const lastGraphErrorRef = useRef(0);
  const backendDownRef = useRef(false);
  const activeToolRef = useRef("");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeFeedback]);

  useEffect(() => {
    if (activeFeedback) {
      setTimeout(() => feedbackTextareaRef.current?.focus(), 0);
    }
  }, [activeFeedback]);

  useEffect(() => {
    return () => {
      pulseTimersRef.current.forEach((timer) => clearTimeout(timer));
      pulseTimersRef.current = [];
      flowTimersRef.current.forEach((timer) => clearTimeout(timer));
      flowTimersRef.current = [];
      if (highlightTimerRef.current) {
        clearTimeout(highlightTimerRef.current);
      }
    };
  }, []);

  const addGraphStep = (text, level = "info") => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setGraphSteps((prev) => [{ id, text, level }, ...prev].slice(0, 6));
  };

  const clearNonFunctionSteps = () => {
    setGraphSteps((prev) => prev.filter((step) => step.text.startsWith("Function:")));
  };

  const setFunctionStep = (label) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setGraphSteps((prev) => {
      const withoutFunction = prev.filter((step) => !step.text.startsWith("Function:"));
      return [{ id, text: `Function: ${label}`, level: "info" }, ...withoutFunction].slice(0, 6);
    });
  };

  const queuePulse = (depth, label, delayMs = 0) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const startTimer = setTimeout(() => {
      setGraphPulses((prev) => [...prev, { id, depth, label, ts: Date.now() }]);
    }, delayMs);
    const endTimer = setTimeout(() => {
      setGraphPulses((prev) => prev.filter((pulse) => pulse.id !== id));
    }, delayMs + 1400);

    pulseTimersRef.current.push(startTimer, endTimer);
  };

  const triggerGraphSearch = (label, traceId) => {
    setGraphMeta((prev) => ({
      ...prev,
      status: "active",
      activeTool: label,
      traceId: traceId || prev.traceId,
      lastEvent: Date.now(),
    }));

    queuePulse(0, label, 0);
    queuePulse(1, label, 220);
    queuePulse(2, label, 440);
  };

  const resetGraphHighlights = () => {
    flowTimersRef.current.forEach((timer) => clearTimeout(timer));
    flowTimersRef.current = [];

    if (highlightTimerRef.current) {
      clearTimeout(highlightTimerRef.current);
      highlightTimerRef.current = null;
    }

    setSearchDepth(-1);
    setHighlightedNodeIds([]);
    setHighlightedEdges([]);
  };

  const animateFlowFromSeed = (nodeIdsRaw, edgesRaw = []) => {
    const nodeIds = (Array.isArray(nodeIdsRaw) ? nodeIdsRaw : []).map(normalizeGraphId);
    const edges = (Array.isArray(edgesRaw) ? edgesRaw : []).map((edge) => ({
      from: normalizeGraphId(edge.from),
      to: normalizeGraphId(edge.to),
      type: edge.type,
    }));

    if (!nodeIds.length) return;

    flowTimersRef.current.forEach((timer) => clearTimeout(timer));
    flowTimersRef.current = [];

    const seed = nodeIds[0];
    const adjacency = new Map();
    edges.forEach((edge) => {
      if (!adjacency.has(edge.from)) adjacency.set(edge.from, new Set());
      if (!adjacency.has(edge.to)) adjacency.set(edge.to, new Set());
      adjacency.get(edge.from).add(edge.to);
      adjacency.get(edge.to).add(edge.from);
    });

    const bfsOrder = [seed];
    const visited = new Set([seed]);
    const queue = [seed];

    while (queue.length) {
      const current = queue.shift();
      const neighbors = [...(adjacency.get(current) || new Set())];
      neighbors.forEach((nb) => {
        if (!visited.has(nb)) {
          visited.add(nb);
          bfsOrder.push(nb);
          queue.push(nb);
        }
      });
    }

    nodeIds.forEach((id) => {
      if (!visited.has(id)) {
        visited.add(id);
        bfsOrder.push(id);
      }
    });

    setHighlightedNodeIds([]);
    setHighlightedEdges([]);
    setSearchDepth(0);
    setGraphMeta((prev) => ({ ...prev, status: "active", currentDepth: 0, lastEvent: Date.now() }));

    bfsOrder.forEach((nodeId, idx) => {
      const timer = setTimeout(() => {
        setSearchDepth(idx);
        setGraphMeta((prev) => ({ ...prev, status: "active", currentDepth: idx, lastEvent: Date.now() }));
        setHighlightedNodeIds((prev) => {
          const merged = new Set(prev);
          merged.add(nodeId);
          return [...merged];
        });
        setHighlightedEdges((prev) => {
          const highlightedNodes = new Set([...highlightedSetFromState(prev), nodeId]);
          const seen = new Set(prev.map((e) => `${e.from}|${e.to}|${e.type || ""}`));
          const merged = [...prev];
          edges.forEach((edge) => {
            if (!(highlightedNodes.has(edge.from) && highlightedNodes.has(edge.to))) return;
            const key = `${edge.from}|${edge.to}|${edge.type || ""}`;
            if (!seen.has(key)) {
              seen.add(key);
              merged.push(edge);
            }
          });
          return merged;
        });
        queuePulse(idx, `flow-${idx}`, 0);
      }, idx * 460);
      flowTimersRef.current.push(timer);
    });

    const finishTimer = setTimeout(() => {
      setSearchDepth(-1);
      setGraphMeta((prev) => ({ ...prev, status: "idle", currentDepth: -1, lastEvent: Date.now() }));
    }, bfsOrder.length * 460 + 500);
    flowTimersRef.current.push(finishTimer);
  };

  const highlightedSetFromState = (edgesState) => {
    const nodes = new Set();
    edgesState.forEach((edge) => {
      nodes.add(normalizeGraphId(edge.from));
      nodes.add(normalizeGraphId(edge.to));
    });
    highlightedNodeIds.forEach((id) => nodes.add(normalizeGraphId(id)));
    return nodes;
  };

  const handleBackendEvent = (evt) => {
    if (!evt || !evt.event) return;
    const payload = evt.payload || {};
    const isGraphToolActive = GRAPH_SEARCH_TOOLS.has(activeToolRef.current);

    if (evt.event === "function_selected") {
      const label = payload.selected_function || "?";
      setFunctionStep(label);
      setGraphMeta((prev) => ({
        ...prev,
        traceId: payload.trace_id || prev.traceId,
        lastEvent: Date.now(),
      }));
      return;
    }

    if (evt.event === "queue_step_start") {
      const tool = payload.tool || "tool";
      clearNonFunctionSteps();
      activeToolRef.current = tool;
      setGraphMeta((prev) => ({
        ...prev,
        activeTool: tool,
        traceId: payload.trace_id || prev.traceId,
        lastEvent: Date.now(),
      }));

      if (GRAPH_SEARCH_TOOLS.has(tool)) {
        // Limpiar highlights anteriores al iniciar nueva búsqueda
        resetGraphHighlights();
        triggerGraphSearch(tool, payload.trace_id);
      } else {
        // Para tools no relacionadas con grafo, no mostrar nodos resaltados.
        resetGraphHighlights();
        setGraphMeta((prev) => ({ ...prev, status: "idle", currentDepth: -1, lastEvent: Date.now() }));
      }
      return;
    }

    if (evt.event === "queue_step_done") {
      return;
    }

    if (evt.event === "queue_step_error") {
      setGraphMeta((prev) => ({ ...prev, status: "idle" }));
      return;
    }

    if (evt.event === "queue_execution_finished") {
      setGraphMeta((prev) => ({ ...prev, status: "idle" }));
    }

    // Evento de paso de búsqueda BFS (nuevo - animación progresiva)
    if (evt.event === "graph_search_step") {
      if (!isGraphToolActive) {
        return;
      }
      const depth = payload.depth ?? 0;
      const nodeIds = Array.isArray(payload.node_ids)
        ? payload.node_ids.map(normalizeGraphId)
        : [];
      const edges = Array.isArray(payload.edges)
        ? payload.edges.map((edge) => ({
            from: normalizeGraphId(edge.from),
            to: normalizeGraphId(edge.to),
          }))
        : [];
      const cumulativeNodes = Array.isArray(payload.cumulative_nodes)
        ? payload.cumulative_nodes.map(normalizeGraphId)
        : nodeIds;
      
      setSearchDepth(depth);
      setHighlightedNodeIds(cumulativeNodes);
      setHighlightedEdges((prev) => {
        const seen = new Set(prev.map((edge) => `${edge.from}|${edge.to}`));
        const merged = [...prev];
        edges.forEach((edge) => {
          const key = `${edge.from}|${edge.to}`;
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(edge);
          }
        });
        return merged;
      });
      setGraphMeta((prev) => ({
        ...prev,
        status: "active",
        currentDepth: depth,
        lastEvent: Date.now(),
      }));
      
      // Disparar pulso visual para este nivel
      queuePulse(depth, `depth-${depth}`, 0);
      return;
    }
    
    // Evento de búsqueda completada
    if (evt.event === "graph_search_complete") {
      if (!isGraphToolActive) {
        return;
      }
      const nodeIds = Array.isArray(payload.node_ids)
        ? payload.node_ids.map(normalizeGraphId)
        : [];
      const edges = Array.isArray(payload.edges)
        ? payload.edges.map((edge) => ({
            from: normalizeGraphId(edge.from),
            to: normalizeGraphId(edge.to),
          }))
        : [];
      
      animateFlowFromSeed(nodeIds, edges);
      
      // Limpiar highlights después de un tiempo
      if (highlightTimerRef.current) {
        clearTimeout(highlightTimerRef.current);
      }
      highlightTimerRef.current = setTimeout(() => {
        setHighlightedNodeIds([]);
        setHighlightedEdges([]);
      }, 8000);
      return;
    }

    // Evento legacy (por compatibilidad)
    if (evt.event === "graph_search_result") {
      if (!isGraphToolActive) {
        return;
      }
      const nodeIds = Array.isArray(payload.node_ids)
        ? payload.node_ids.map(normalizeGraphId)
        : [];
      setHighlightedNodeIds(nodeIds);
      if (highlightTimerRef.current) {
        clearTimeout(highlightTimerRef.current);
      }
      highlightTimerRef.current = setTimeout(() => {
        setHighlightedNodeIds([]);
        setHighlightedEdges([]);
      }, 6000);
    }
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/events/${USER_ID}`, {
          params: { limit: 100 },
        });
        const events = res.data?.events || [];
        events.forEach((evt) => {
          handleBackendEvent(evt);
        });
        if (backendDownRef.current) {
          backendDownRef.current = false;
          clearNonFunctionSteps();
        }
      } catch (error) {
        const now = Date.now();
        if (now - lastEventsErrorRef.current > 8000) {
          console.error("[BACKEND][events] polling_error", error?.message || error);
          lastEventsErrorRef.current = now;
        }
        if (!backendDownRef.current) {
          backendDownRef.current = true;
          clearNonFunctionSteps();
        }
      }
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/graph`, {
          params: { limit_nodes: 60, limit_edges: 120, user_id: USER_ID },
        });
        const nodes = res.data?.nodes || [];
        const edges = res.data?.edges || [];
        setGraphData({ nodes, edges });
      } catch (error) {
        const now = Date.now();
        if (now - lastGraphErrorRef.current > 8000) {
          console.error("[FRONTEND][graph] polling_error", error?.message || error);
          lastGraphErrorRef.current = now;
        }
      }
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    setActiveFeedback(null);
    setCommentText("");

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        query: userMessage.content,
        user_id: USER_ID,
      });

      const data = response.data;

      const botMessage = {
        role: "bot",
        content: data.response,
        run_id: data.run_id,
        feedbackGiven: false,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("[FRONTEND] chat_error", error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: "⚠️ Error de conexión con el servidor.", isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (runId, score, index, customComment = null) => {
    try {
      const cleanComment = (customComment ?? "").trim();
      const finalComment =
        score === 1 ? "Usuario satisfecho" : cleanComment || "Error reportado";

      await axios.post(`${API_URL}/feedback`, {
        run_id: runId,
        score,
        comment: finalComment,
      });

      setMessages((prev) =>
        prev.map((msg, i) =>
          i === index ? { ...msg, feedbackGiven: true, feedbackScore: score } : msg
        )
      );

      setActiveFeedback(null);
      setCommentText("");

      if (score === 0 && cleanComment.length >= 5) {
        alert("¡Gracias! He guardado tu corrección en mi memoria para la próxima vez.");
      }
    } catch (error) {
      console.error("[FRONTEND] feedback_error", error);
    }
  };

  const handleThumbsDown = (runId, index) => {
    setActiveFeedback({ runId, index });
  };

  const cancelFeedback = () => {
    setActiveFeedback(null);
    setCommentText("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const graphLayout = useMemo(() => {
    const width = GRAPH_CANVAS.width;
    const height = GRAPH_CANVAS.height;
    const nodes = graphData.nodes || [];
    if (!nodes.length) {
      return { width, height, nodes: [], edges: [] };
    }
    const radius = Math.min(width, height) / 2 - 22;
    const centerX = width / 2;
    const centerY = height / 2;
    const positionedNodes = nodes.map((node, idx) => {
      const angle = (Math.PI * 2 * idx) / nodes.length - Math.PI / 2;
      return {
        ...node,
        x: Math.round(centerX + Math.cos(angle) * radius),
        y: Math.round(centerY + Math.sin(angle) * radius),
      };
    });
    const nodeMap = new Map(positionedNodes.map((node) => [normalizeGraphId(node.id), node]));
    const edges = (graphData.edges || []).filter(
      (edge) => nodeMap.has(normalizeGraphId(edge.from)) && nodeMap.has(normalizeGraphId(edge.to))
    );
    return { width, height, nodes: positionedNodes, edges };
  }, [graphData]);

  const isScanning = graphMeta.status === "active" && graphPulses.length > 0;
  const highlightedSet = useMemo(() => new Set(highlightedNodeIds), [highlightedNodeIds]);
  const highlightedEdgeSet = useMemo(() => {
    const set = new Set();
    highlightedEdges.forEach((e) => {
      const from = normalizeGraphId(e.from);
      const to = normalizeGraphId(e.to);
      set.add(`${from}|${to}`);
      set.add(`${to}|${from}`); // bidireccional
    });
    return set;
  }, [highlightedEdges]);

  return (
    <div className="app-container">
      <header className="chat-header">
        <h1>🛍️ Agente E-Commerce AI</h1>
        <span className="status-badge">Online</span>
      </header>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message-row ${msg.role}`}>
            <div className="avatar">
              {msg.role === "bot" ? <Bot size={24} /> : <User size={24} />}
            </div>

            <div className="message-column">
              <div className={`message-bubble ${msg.isError ? "error" : ""}`}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>

              {msg.role === "bot" && msg.run_id && (
                <div className="feedback-section">
                  {!msg.feedbackGiven ? (
                    <div className="feedback-actions">
                      <span className="feedback-label">¿Útil?</span>
                      <button
                        onClick={() => sendFeedback(msg.run_id, 1, index)}
                        className="btn-icon up"
                        aria-label="Útil"
                      >
                        <ThumbsUp size={14} />
                      </button>

                      <button
                        onClick={() => handleThumbsDown(msg.run_id, index)}
                        className="btn-icon down"
                        aria-label="No útil"
                      >
                        <ThumbsDown size={14} />
                      </button>
                    </div>
                  ) : (
                    <span className="feedback-sent">
                      {msg.feedbackScore === 1 ? "✅ Feedback Positivo" : "📝 Corrección Enviada"}
                    </span>
                  )}
                </div>
              )}

              {activeFeedback && activeFeedback.index === index && !msg.feedbackGiven && (
                <div className="feedback-form">
                  <p className="feedback-instruction">¿Qué dato está mal? (Ayúdame a aprender):</p>
                  <textarea
                    ref={feedbackTextareaRef}
                    className="feedback-input"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Ej: El precio de la MacBook es 900, no 1200..."
                    rows={2}
                  />
                  <div className="feedback-buttons">
                    <button className="btn-cancel" onClick={cancelFeedback}>
                      Cancelar
                    </button>
                    <button
                      className="btn-submit"
                      onClick={() => sendFeedback(msg.run_id, 0, index, commentText)}
                      disabled={commentText.trim().length < 5}
                    >
                      Enviar Corrección <Send size={12} style={{ marginLeft: 6 }} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-row bot">
            <div className="avatar">
              <Bot size={24} />
            </div>
            <div className="message-bubble loading">
              <span className="dot">.</span>
              <span className="dot">.</span>
              <span className="dot">.</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Pregunta por laptops, precios o corrígeme..."
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            <ArrowUp size={20} strokeWidth={2} />
          </button>
        </div>
      </div>

      <aside className="graph-overlay" aria-live="polite">
        <div className="graph-card">
          <div className="graph-header">
            <div className="graph-title">Grafo en vivo</div>
            <div className={`graph-status ${graphMeta.status}`}>
              {graphMeta.status === "active" ? "Buscando" : "En espera"}
            </div>
          </div>

          <svg
            className="graph-canvas"
            viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
            role="img"
            aria-label="Visualizacion de busqueda en el grafo"
          >
            {graphLayout.edges.map((edge) => {
              const isActive = isScanning;
              const edgeFrom = normalizeGraphId(edge.from);
              const edgeTo = normalizeGraphId(edge.to);
              const edgeKey = `${edgeFrom}|${edgeTo}`;
              const isEdgeHighlighted = highlightedEdgeSet.has(edgeKey);
              const isNodeHighlighted = highlightedSet.has(edgeFrom) || highlightedSet.has(edgeTo);
              const source = graphLayout.nodes.find((n) => normalizeGraphId(n.id) === edgeFrom);
              const target = graphLayout.nodes.find((n) => normalizeGraphId(n.id) === edgeTo);
              if (!source || !target) return null;
              return (
                <line
                  key={`${edgeFrom}-${edgeTo}`}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  className={`graph-edge ${isActive ? "active" : ""} ${isEdgeHighlighted ? "highlight-edge" : ""} ${isNodeHighlighted ? "highlight" : ""}`}
                />
              );
            })}

            {graphLayout.nodes.map((node, idx) => {
              const isActive = isScanning;
              const isHighlighted = highlightedSet.has(normalizeGraphId(node.id));
              const isSeed = idx === 0 || (searchDepth === 0 && isHighlighted);
              return (
                <g 
                  key={node.id} 
                  className={`graph-node ${isActive ? "active" : ""} ${isHighlighted ? "highlight" : ""} ${isSeed && isHighlighted ? "seed" : ""}`}
                >
                  <circle cx={node.x} cy={node.y} r={isSeed ? 12 : 8} />
                  {isHighlighted && (
                    <circle 
                      cx={node.x} 
                      cy={node.y} 
                      r={isSeed ? 16 : 12} 
                      className="pulse-ring" 
                    />
                  )}
                  <title>{node.label || "Nodo"}</title>
                </g>
              );
            })}
          </svg>

          <div className="graph-meta">
            <div className="graph-meta-row">
              <span className="meta-label">Tool</span>
              <span className="meta-value">{graphMeta.activeTool || "-"}</span>
            </div>
            <div className="graph-meta-row">
              <span className="meta-label">Depth</span>
              <span className="meta-value depth-indicator">
                {graphMeta.currentDepth >= 0 ? `🔍 ${graphMeta.currentDepth}` : "-"}
              </span>
            </div>
            <div className="graph-meta-row">
              <span className="meta-label">Nodos</span>
              <span className="meta-value">
                {highlightedNodeIds.length > 0 
                  ? `${highlightedNodeIds.length} / ${graphData.nodes.length}` 
                  : graphData.nodes.length}
              </span>
            </div>
          </div>
        </div>

        <div className="graph-log">
          {graphSteps.length === 0 && (
            <div className="graph-log-empty">Function: -</div>
          )}
          {graphSteps.map((step) => (
            <div key={step.id} className={`graph-log-item ${step.level}`}>
              {step.text}
            </div>
          ))}
        </div>

      </aside>
    </div>
  );
}

export default App;
