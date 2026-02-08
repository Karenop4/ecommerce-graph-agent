import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { ArrowUp, User, Bot, ThumbsUp, ThumbsDown, Send } from "lucide-react";
import "./App.css";

const API_URL = "http://localhost:8000";
const USER_ID = "usuario_demo";

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

  const [activeFeedback, setActiveFeedback] = useState(null); // { runId, index }
  const [commentText, setCommentText] = useState("");

  const messagesEndRef = useRef(null);
  const feedbackTextareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeFeedback]);

  useEffect(() => {
    if (activeFeedback) {
      setTimeout(() => feedbackTextareaRef.current?.focus(), 0);
    }
  }, [activeFeedback]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/events/${USER_ID}`, {
          params: { limit: 100 },
        });
        const events = res.data?.events || [];
        events.forEach((evt) => {
          console.log(
            `[BACKEND][${evt.level || "info"}] ${evt.event || "event"}`,
            evt.payload || {}
          );
        });
      } catch (error) {
        console.error("[BACKEND][events] polling_error", error?.message || error);
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
      console.log("[FRONTEND] sending_chat", { query: userMessage.content, user_id: USER_ID });
      const response = await axios.post(`${API_URL}/chat`, {
        query: userMessage.content,
        user_id: USER_ID,
      });

      const data = response.data;
      console.log("[FRONTEND] chat_response", data);

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

      console.log("[FRONTEND] sending_feedback", { runId, score, finalComment });
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
    </div>
  );
}

export default App;
