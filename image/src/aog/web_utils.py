import networkx as nx
import mpld3, mpld3.utils, mpld3.plugins

import matplotlib, matplotlib.lines
import matplotlib.pyplot as plt


class HighlightPlugin(mpld3.plugins.PluginBase):
    """A simple plugin showing how multiple axes can be linked"""

    with open("./aog/static/javascript/highlight_plugin.js", "r") as f:
        JAVASCRIPT = f.read()

    with open("./aog/static/css/remove_axis.css") as f:
        CSS = f.read()

    def __init__(self, node_bboxes, node_texts, edges,
                 highlight_edges, highlight_colors, default_color="black"):
        self.css_ = self.CSS
        self.dict_ = {
            "type": "highlight",
            "id_node_bboxes": [mpld3.utils.get_id(bbox) for bbox in node_bboxes],
            "id_node_texts": [mpld3.utils.get_id(text) for text in node_texts],
            "id_edges": [mpld3.utils.get_id(edge) for edge in edges],
            "highlight_edges": highlight_edges,
            "highlight_colors": highlight_colors,
            "default_color": default_color,
        }