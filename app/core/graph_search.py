from __future__ import annotations

from typing import Any


def buscar_en_grafo(driver: Any, termino: str, limite: int = 5, max_depth: int = 2) -> list[dict[str, Any]]:
    """
    Búsqueda en grafo con estrategia BFS acotada por profundidad:
    1) Encuentra nodos semilla por match textual de propiedades.
    2) Expande vecinos 1..max_depth y devuelve paths/nodos relacionados.
    """
    if not termino.strip():
        return []

    cypher = """
    MATCH (seed)
    WHERE any(prop IN keys(seed)
      WHERE toString(seed[prop]) IS NOT NULL
      AND toLower(toString(seed[prop])) CONTAINS toLower($termino)
    )
    WITH seed LIMIT $limite
    MATCH p=(seed)-[*1..$max_depth]-(related)
    RETURN
      labels(seed) AS seed_labels,
      properties(seed) AS seed_props,
      labels(related) AS related_labels,
      properties(related) AS related_props,
      length(p) AS depth
    ORDER BY depth ASC
    LIMIT $limite
    """
    with driver.session() as session:
        return [
            dict(r)
            for r in session.run(
                cypher,
                termino=termino,
                limite=limite,
                max_depth=max_depth,
            )
        ]


def format_graph_search_results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No encontré nodos/relaciones para ese término en el grafo."

    lines = ["Resultados de búsqueda BFS en el grafo:"]
    for idx, row in enumerate(rows, start=1):
        seed_labels = ",".join(row.get("seed_labels", []))
        related_labels = ",".join(row.get("related_labels", []))
        seed_props = row.get("seed_props", {})
        related_props = row.get("related_props", {})
        depth = row.get("depth", "?")
        lines.append(
            f"{idx}) depth={depth} | seed=[{seed_labels}] {seed_props} -> related=[{related_labels}] {related_props}"
        )
    return "\n".join(lines)
