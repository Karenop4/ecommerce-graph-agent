from __future__ import annotations

import time
from typing import Any, Callable, Optional


def buscar_en_grafo(
    driver: Any,
    termino: str,
    limite: int = 5,
    max_depth: int = 2,
    on_step: Optional[Callable[[int, list, list], None]] = None,
    step_delay: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Búsqueda en grafo con estrategia BFS acotada por profundidad:
    1) Encuentra nodos semilla por match textual de propiedades.
    2) Expande vecinos 1..max_depth y devuelve paths/nodos relacionados.
    
    Args:
        on_step: Callback llamado en cada nivel del BFS con (depth, node_ids, edge_tuples)
        step_delay: Delay entre pasos para visualización (segundos)
    """
    if not termino.strip():
        return []

    results = []
    
    with driver.session() as session:
        # Paso 1: Encontrar nodos semilla
        cypher_seeds = """
        MATCH (seed)
        WHERE any(prop IN keys(seed)
          WHERE toString(seed[prop]) IS NOT NULL
          AND toLower(toString(seed[prop])) CONTAINS toLower($termino)
        )
        RETURN elementId(seed) AS seed_id,
               labels(seed) AS seed_labels,
               properties(seed) AS seed_props
        LIMIT $limite
        """
        seed_rows = list(session.run(cypher_seeds, termino=termino, limite=limite))
        seed_ids = [r["seed_id"] for r in seed_rows]
        
        if not seed_ids:
            return []
        
        # Emitir paso 0: nodos semilla encontrados
        if on_step:
            on_step(0, seed_ids, [])
            time.sleep(step_delay)
        
        # Paso 2: Expandir por cada nivel de profundidad
        visited_nodes = set(seed_ids)
        current_frontier = seed_ids
        
        for depth in range(1, max_depth + 1):
            if not current_frontier:
                break
                
            # Obtener vecinos del nivel actual
            cypher_expand = """
            MATCH (n)-[r]-(neighbor)
            WHERE elementId(n) IN $frontier_ids
            RETURN DISTINCT
                   elementId(n) AS from_id,
                   elementId(neighbor) AS to_id,
                   labels(neighbor) AS neighbor_labels,
                   properties(neighbor) AS neighbor_props,
                   type(r) AS rel_type
            LIMIT $limite
            """
            expand_rows = list(session.run(
                cypher_expand,
                frontier_ids=current_frontier,
                limite=limite * 3
            ))
            
            new_node_ids = []
            edge_tuples = []
            
            for row in expand_rows:
                to_id = row["to_id"]
                from_id = row["from_id"]
                edge_tuples.append((from_id, to_id, row["rel_type"]))
                
                if to_id not in visited_nodes:
                    visited_nodes.add(to_id)
                    new_node_ids.append(to_id)
                    
                    # Encontrar el seed original para este nodo
                    for seed_row in seed_rows:
                        results.append({
                            "seed_id": seed_row["seed_id"],
                            "seed_labels": seed_row["seed_labels"],
                            "seed_props": seed_row["seed_props"],
                            "related_id": to_id,
                            "related_labels": row["neighbor_labels"],
                            "related_props": row["neighbor_props"],
                            "depth": depth,
                        })
                        break  # Solo asociar con el primer seed
            
            # Emitir paso del nivel actual
            if on_step and (new_node_ids or edge_tuples):
                all_level_nodes = list(current_frontier) + new_node_ids
                on_step(depth, all_level_nodes, edge_tuples)
                time.sleep(step_delay)
            
            current_frontier = new_node_ids
            
            if len(results) >= limite:
                break
    
    return results[:limite]


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
