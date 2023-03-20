import graphviz
import os
os.environ["PATH"] += ":/home/limingjie/usr/bin"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout, pydot_layout
import numpy as np
import random
import matplotlib.patches as patches


class AOGNode(object):
    def __init__(self, type, name="anonymous", label=None, value=None, layer_id=-1, children=None):
        '''
        To create a node in an AOG, we have to specify:
          (1) The node type: is it an AND node? is it a 'plus' node (on the top of the AOG)? is it a leaf node?
          (2) The name of the node: The id that uniquely determines a node in an AOG.
              [WARNING!!!!!] All nodes need to have different names.
          (3) The label of the node: when visualizing the graph, the nodes' label. Different nodes can share the same label.
          (4) The value of the node (if any): the interaction
          (5) The layer_id of the node: This node is in the ?-th layer.
          (6) The children of this node. This do not need to be pre-defined, as its corresponding children can be set later.
        :param type: The type of the node. Supported types: AND, +, leaf
        :param name: The unique id of each node in a graph
        :param children: The children of the current node.
        '''
        # set the type of the node
        assert type in ["AND", "+", "leaf"]
        self.type = type
        # set the name
        assert name != "anonymous"
        self.name = name
        # set the label
        self.label = label if label is not None else name
        # set the value (if any)
        self.value = value
        # set the layer id
        assert isinstance(layer_id, int) and layer_id > 0
        self.layer_id = layer_id

        self.children = None
        if children is not None:
            self.extend_children(children)

    def extend_children(self, children: list):
        assert len(children) > 0
        if self.children is None:
            self.children = []
        for child in children:
            if child not in self.children: self.children.append(child)

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        return f"<AOG Node> at {id(self)}: type={self.type}, name={self.name}"


class AOG(object):
    def __init__(self, root: AOGNode=None):
        '''
        To construct an AOG, we only need to specify its root node.
        :param root: The root node of the AOG.
        '''
        self.root = root

    def visualize(self, save_path, renderer="graphviz", **kwargs):
        '''
        Use this method to visualize the current AOG.
        :param save_path: The save path of the visualized graph.
        :param renderer: The renderer. Now supported: graphviz, networkx
        :param kwargs: Other parameters. TBD later.
        :return: (void)
        '''
        random.seed(0)
        if renderer == "graphviz":
            g = graphviz.Graph('AOG', format="svg", strict=True)
            g.graph_attr.update(ratio="0.3")
            if self.root is None:
                g.render(save_path)
                return
            g.node(self.root.name, label=self.root.type)
            self._add_node_to_graphviz(g, self.root)
            g.render(save_path)
        elif renderer == "networkx":
            g = nx.DiGraph()
            if "figsize" in kwargs.keys():
                plt.figure(figsize=kwargs["figsize"])
            else:
                plt.figure(figsize=(20, 8))
            if self.root is None:
                nx.draw(g)
                plt.axis("off")
                plt.savefig(save_path, dpi=200)
                return
            # ================================================
            # start constructing the Graph object in networkx
            # ================================================
            g.add_node(self.root.name, label=self.root.label, layer_id=self.root.layer_id, value=self.root.value)
            self._add_node_to_networkx(g, self.root)

            self._generate_aog_hierarchy_networkx(g)

            pos = self._generate_layout_networkx(g, **kwargs)

            node_attr = self._generate_node_attr_networkx(g, **kwargs)
            nx.draw_networkx_nodes(g, pos, **node_attr)
            edge_attr = self._generate_edge_attr_networkx(g, **kwargs)
            nx.draw_networkx_edges(g, pos, **edge_attr)
            label_attr = self._generate_label_attr_networkx(g, **kwargs)
            nx.draw_networkx_labels(g, pos, labels=dict(g.nodes.data("label")), **label_attr)
            self._add_annotation_networkx(plt.gca(), **kwargs)
            title = self._generate_title_networkx(g, **kwargs)
            plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'})
            plt.xlim(-0.1, 1.1)
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(save_path, dpi=200)

    def _add_node_to_graphviz(self, g, node):
        for child in node.children:
            if child.type == "leaf":
                g.node(child.name)
                g.edge(node.name, child.name)
            else:
                # g.node(child.name, label=child.type)
                g.node(child.name)
                g.edge(node.name, child.name)
                self._add_node_to_graphviz(g, child)

    def _add_node_to_networkx(self, g, node):
        for child in node.children:
            if child.type == "leaf":
                if not g.has_node(child.name):
                    g.add_node(child.name, label=child.label, layer_id=child.layer_id, value=child.value)
                if not g.has_edge(node.name, child.name):
                    g.add_edge(node.name, child.name)
            else:
                if not g.has_node(child.name):
                    g.add_node(child.name, label=child.label, layer_id=child.layer_id, value=child.value)
                if not g.has_edge(node.name, child.name):
                    g.add_edge(node.name, child.name)
                self._add_node_to_networkx(g, child)

    def _generate_aog_hierarchy_networkx(self, g):
        assert '+' in g.nodes
        layer_4 = ['+']
        layer_3 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 3]
        layer_2 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 2]
        layer_1 = [node_name for node_name in g.nodes if g.nodes[node_name]["layer_id"] == 1]
        self.layer_4 = layer_4
        self.layer_3 = layer_3
        self.layer_2 = layer_2
        self.layer_1 = layer_1

    def _generate_layout_networkx(self, g, **kwargs):
        assert '+' in g.nodes
        if 'n_row_interaction' in kwargs.keys(): n_row_interaction = kwargs["n_row_interaction"]
        else: n_row_interaction = 2

        layer_4 = self.layer_4
        layer_3 = self.layer_3
        layer_2 = self.layer_2
        layer_1 = self.layer_1
        random.shuffle(layer_3)
        pos = {}
        # set the position of layer-1 nodes
        x = np.linspace(0, 1, len(layer_1))
        for i in range(len(layer_1)):
            pos[layer_1[i]] = np.array([x[i], 0])
        # set the position of layer-2 nodes
        x = np.linspace(0.07, 0.93, len(layer_2))
        y = np.linspace(0, 1, n_row_interaction + 3)[1]
        for i in range(len(layer_2)):
            pos[layer_2[i]] = np.array([x[i], y])
        # set the position of layer-3 nodes
        n_col_interaction = int(np.ceil(len(layer_3) / n_row_interaction))
        x = np.linspace(0.02, 0.98, n_col_interaction)
        y = np.linspace(0, 1, n_row_interaction + 3)[2:-1][::-1]
        for i in range(len(layer_3)):
            if i // n_col_interaction == n_row_interaction - 1 and i % n_col_interaction == 0:
                n_rest = n_col_interaction * n_row_interaction - len(layer_3)
                x += 0.5 * n_rest * (x[1] - x[0])
            pos[layer_3[i]] = np.array([x[i%n_col_interaction], y[i//n_col_interaction]])
        # set the position of layer-4 nodes
        pos[layer_4[0]] = np.array([0.5, 1])
        return pos

    def _generate_node_attr_networkx(self, g, **kwargs):
        min_alpha = 0.05
        # colors = ["#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]  # layer-4 to layer-1
        colors = ["#9e9e9e", {'pos': "#e53935", 'neg': "#1e88e5"}, "#ce93d8", "#9e9e9e"]  # layer-4 to layer-1  # positive interaction is red
        if "reverse_color" in kwargs and kwargs["reverse_color"] == True:
            colors = ["#9e9e9e", {'neg': "#e53935", 'pos': "#1e88e5"}, "#ce93d8", "#9e9e9e"]  # layer-4 to layer-1  # positive interaction is blue
        node_attr = {
            "node_size": [],
            "node_color": [],
            "alpha": []
        }
        max_value = max([abs(g.nodes[node_name]['value']) for node_name in self.layer_3])
        for node in g.nodes:
            node_attr["node_size"].append(3000)
            if node in self.layer_4: node_attr["node_color"].append(colors[0])
            elif node in self.layer_3:
                if g.nodes[node]['value'] > 0: node_attr["node_color"].append(colors[1]['pos'])
                else: node_attr["node_color"].append(colors[1]['neg'])
            elif node in self.layer_2: node_attr["node_color"].append(colors[2])
            elif node in self.layer_1: node_attr["node_color"].append(colors[3])
            else: raise Exception
            if node in self.layer_3:
                alpha = abs(g.nodes[node]['value']) / max_value
                alpha = alpha * (1 - min_alpha) + min_alpha
                node_attr["alpha"].append(alpha)
            else:
                node_attr["alpha"].append(0.8)
        return node_attr

    def _generate_edge_attr_networkx(self, g, **kwargs):
        highlight_path = None
        if "highlight_path" in kwargs.keys(): highlight_path = kwargs["highlight_path"]
        highlight_edges = []
        if highlight_path == "max":
            target_node = sorted(self.layer_3, key=lambda x: abs(float(g.nodes[x]["value"])), reverse=True)[0]
            highlight_edges.append((self.root.name, target_node))
            highlight_edges.extend(nx.bfs_edges(g, target_node))
        edge_attr = {
            "alpha": 0.4,
            "width": [],
            "edge_color": []
        }

        for u, v in g.edges:
            if (u, v) in highlight_edges or (v, u) in highlight_edges:
                edge_attr["width"].append(6)
                edge_attr["edge_color"].append("red")
            else:
                edge_attr["width"].append(3)
                edge_attr["edge_color"].append("gray")

        return edge_attr

    def _generate_label_attr_networkx(self, g, **kwargs):
        return {
            "font_size": 14,
            "bbox": {"ec": "white", "fc": "white", "alpha": 0.7}
        }

    def _add_annotation_networkx(self, ax, **kwargs):
        if 'n_row_interaction' in kwargs.keys(): n_row_interaction = kwargs["n_row_interaction"]
        else: n_row_interaction = 2
        y = np.linspace(0, 1, n_row_interaction + 3)
        interval = y[1] - y[0]
        rect = patches.Rectangle(
            (-0.01, y[2] - 0.3 * interval),
            1.02, y[-2] - y[2] + 0.6 * interval,
            linewidth=5, linestyle="dashed",
            edgecolor='lightsteelblue', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(-0.03, (y[-2] + y[2]) / 2, "The Most Salient Patterns",
                rotation=90, va="center", fontsize=15, color="lightsteelblue", weight="bold")

    def _generate_title_networkx(self, g, **kwargs):
        title = ""
        if "title" in kwargs.keys():
            title = kwargs["title"]
        title += f" | # edges: {g.number_of_edges()}"
        title += f" | # nodes: {g.number_of_nodes()}"
        return title



if __name__ == '__main__':

    A = AOGNode(type="leaf", name="A", children=None)
    B = AOGNode(type="leaf", name="B", children=None)
    C = AOGNode(type="leaf", name="C", children=None)
    D = AOGNode(type="leaf", name="D", children=None)
    E = AOGNode(type="leaf", name="E", children=None)

    aog = AOG(root=AOGNode(
        type="OR", name="AB(C+D)+DE",
        children=[
            AOGNode(type="AND", name="AB(C+D)", children=[
                AOGNode(type="AND", name="AB", children=[A, B]),
                AOGNode(type="+", name="(C+D)", children=[C, D])
            ]),
            AOGNode(type="AND", name="DE", children=[D, E])
        ]
    ))

    aog.visualize("test")