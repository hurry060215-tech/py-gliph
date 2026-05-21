"""Network visualisation of GLIPH / GLIPH2 specificity groups.

A matplotlib + networkx replacement for turboGliph's visNetwork-based
``plot_network``.
"""
from __future__ import annotations

from typing import Iterable

__all__ = ["plot_network"]


def _build_graph(connections):
    import networkx as nx
    g = nx.Graph()
    for _, row in connections.iterrows():
        v1, v2, typ = row.iloc[0], row.iloc[1], row.iloc[2]
        if typ == "singleton":
            g.add_node(v1)
        else:
            g.add_edge(v1, v2, type=typ)
    return g


def plot_network(
    clustering_output: dict,
    show_singletons: bool = False,
    only_clusters: Iterable[str] | None = None,
    layout: str = "spring",
    node_size: int = 120,
    edge_color_map: dict | None = None,
    figsize: tuple = (10, 10),
    seed: int = 42,
    ax=None,
):
    """Draw the clone network coloured by similarity type.

    Parameters
    ----------
    clustering_output
        Output of :func:`pygliph.gliph2` or :func:`pygliph.turbo_gliph`.
    show_singletons
        Whether to draw unconnected sequences.
    only_clusters
        If given, restrict the plot to members of these cluster tags.
    layout
        ``"spring"``, ``"kamada_kawai"`` or ``"circular"``.
    edge_color_map
        Maps edge type (``"local"`` / ``"global"``) to a colour.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the network was drawn on.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    connections = clustering_output["connections"]
    if connections is None or len(connections) == 0:
        raise ValueError("clustering_output has no connections to plot.")

    conn = connections
    if not show_singletons:
        conn = conn[conn.iloc[:, 2] != "singleton"]

    if only_clusters is not None and conn.shape[1] >= 4:
        keep = set(only_clusters)
        conn = conn[conn.iloc[:, 3].isin(keep)]

    g = _build_graph(conn)
    if g.number_of_nodes() == 0:
        raise ValueError("Nothing to plot after filtering.")

    if layout == "spring":
        pos = nx.spring_layout(g, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g)
    elif layout == "circular":
        pos = nx.circular_layout(g)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    if edge_color_map is None:
        edge_color_map = {"local": "#1f77b4", "global": "#d62728",
                          "singleton": "#999999"}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for etype, colour in edge_color_map.items():
        edges = [(u, v) for u, v, d in g.edges(data=True)
                 if d.get("type") == etype]
        if edges:
            nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=colour,
                                   ax=ax, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(g, pos, node_size=node_size,
                           node_color="#ffbb44", edgecolors="#444444", ax=ax)
    ax.set_axis_off()
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=t)
               for t, c in edge_color_map.items()
               if any(d.get("type") == t for _, _, d in g.edges(data=True))]
    if handles:
        ax.legend(handles=handles, title="Similarity", loc="best")
    ax.set_title("GLIPH specificity-group network")
    return ax
