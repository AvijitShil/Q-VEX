"""Structural Graph Builder."""

class StructuralGraphBuilder:
    """Builds NEXT_CHUNK and PREV_CHUNK structural edges for ordered chunks, including 2-hop lookaheads."""

    def build_edges(self, node_ids: list[int]) -> list[tuple[int, int, str, float]]:
        """
        Given a sequential list of node IDs, return the bidirectional structural edges with confidences.
        
        Returns:
            A list of (source_id, target_id, edge_type, confidence)
        """
        edges = []
        n = len(node_ids)
        for i in range(n):
            # 1-hop neighbors (weight = 1.0)
            if i < n - 1:
                edges.append((node_ids[i], node_ids[i + 1], "NEXT_CHUNK", 1.0))
                edges.append((node_ids[i + 1], node_ids[i], "PREV_CHUNK", 1.0))
            
            # 2-hop neighbors (weight = 0.5)
            if i < n - 2:
                edges.append((node_ids[i], node_ids[i + 2], "NEXT_2_CHUNK", 0.5))
                edges.append((node_ids[i + 2], node_ids[i], "PREV_2_CHUNK", 0.5))
                
        return edges
