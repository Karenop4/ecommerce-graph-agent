from __future__ import annotations

from typing import Any


def buscar_en_grafo(driver: Any, termino: str, limite: int = 5) -> list[dict[str, Any]]:
    if not termino.strip():
        return []

    cypher = """
    MATCH (n)
    WHERE any(prop IN keys(n) WHERE toString(n[prop]) IS NOT NULL AND toLower(toString(n[prop])) CONTAINS toLower($termino))
    RETURN labels(n) AS labels, properties(n) AS props
    LIMIT $limite
    """
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, termino=termino, limite=limite)]


def format_graph_search_results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No encontré nodos relacionados en el grafo."
    lines = ["Resultados de búsqueda en el grafo:"]
    for idx, row in enumerate(rows, start=1):
        labels = ",".join(row.get("labels", []))
        props = row.get("props", {})
        lines.append(f"{idx}) labels=[{labels}] props={props}")
    return "\n".join(lines)
