import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import io
import base64
from langgraph.graph import StateGraph
from graphviz import Digraph

from ..config import settings

def visualize_graph(graph: StateGraph, show_state: bool = False, 
                   layout: Optional[str] = None, 
                   highlight_nodes: Optional[List[str]] = None) -> str:
    """
    Visualize a LangGraph as a Graphviz digraph.
    
    Args:
        graph: The LangGraph StateGraph to visualize
        show_state: Whether to show the state details
        layout: The graph layout (dot, neato, fdp, sfdp, twopi, circo)
        highlight_nodes: List of node names to highlight
        
    Returns:
        HTML string with the rendered graph
    """
    # Use config setting if not specified
    if layout is None:
        layout = settings.viz.graph_layout
    
    # Create Graphviz graph
    dot = Digraph(format='svg', engine=layout)
    dot.attr(rankdir='LR', size='8,5', ratio='fill')
    
    # Get the graph from the StateGraph
    nx_graph = graph.graph
    
    # Add nodes
    for node in nx_graph.nodes():
        if node == "END":
            dot.node(node, shape='doublecircle', style='filled', fillcolor='lightblue')
        elif highlight_nodes and node in highlight_nodes:
            dot.node(node, shape='box', style='filled', fillcolor='lightgreen')
        else:
            dot.node(node, shape='box')
    
    # Add edges
    for edge in nx_graph.edges(data=True):
        source, target, data = edge
        
        # Skip conditional edges
        if 'condition' in data:
            continue
            
        dot.edge(source, target)
    
    # Add conditional edges
    for node, edges in graph.conditional_edges.items():
        for condition, target in edges.items():
            if isinstance(target, str):
                dot.edge(node, target, label=condition, style='dashed')
    
    # Render the graph
    graph_svg = dot.pipe(format='svg').decode('utf-8')
    
    return graph_svg
    
def create_interactive_graph(graph: StateGraph) -> None:
    """
    Create an interactive visualization of a LangGraph using matplotlib.
    
    Args:
        graph: The LangGraph StateGraph to visualize
    """
    G = nx.DiGraph()
    
    # Get the graph from the StateGraph
    nx_graph = graph.graph
    
    # Add nodes and edges
    for node in nx_graph.nodes():
        G.add_node(node)
        
    for edge in nx_graph.edges():
        G.add_edge(edge[0], edge[1])
    
    # Add conditional edges
    for node, edges in graph.conditional_edges.items():
        for condition, target in edges.items():
            if isinstance(target, str):
                G.add_edge(node, target, condition=condition)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    edge_labels = {(u, v): d.get('condition', '') for u, v, d in G.edges(data=True) if 'condition' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.axis('off')
    plt.title('LangGraph Visualization')
    plt.tight_layout()
    
    # Save or show
    plt.show() 