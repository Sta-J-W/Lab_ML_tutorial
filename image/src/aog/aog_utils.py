import numpy as np
from .graph import AOGNode, AOG
from pprint import pprint


def get_concept_description(attributes, single_feature):
    '''
    Generate a description of attributes
    :param attributes: basic attributes (in the original dataset) (n_dim,) list
    :param single_feature: Bool mask of a 'single features' (n_dim,)
    :return: a description string
    '''
    description = []
    for i in range(single_feature.shape[0]):
        if single_feature[i]: description.append(attributes[i])
    description = "\n".join(description)
    return description


def generate_interaction_notation(indices, n_dim):
    variables = [r"x_{" + str(i+1) + r"}" for i in range(n_dim)]
    variables += [r"\alpha", r"\beta", r"\gamma", r"\zeta", r"\xi"]

    indices = [variables[i] for i in indices]
    notation = "".join(indices)
    # notation = r"$I($" + notation + r"$)$"
    notation = r"$w_{\{" + notation + r"\}}$"
    # print(notation)
    return notation


def construct_AOG_v1(attributes, attributes_baselines, single_features, concepts, eval_val):
    '''
    Construct a simple And-Or Graph
    :param attributes: basic attributes (in the original dataset) (n_dim,) list
    :param attributes_baselines: basic attributes (arrow compared with their baselines) (n_dim,) list
    :param single_features: Bool masks of 'single features', including the aggregated ones (n_dim', n_dim)
    :param concepts: Bool masks representing extracted concepts (n_concept, n_dim')
    :param eval_val: The multi-variate interaction of these concepts (n_concept,)
    :return: an AOG
    '''
    single_feature_nodes = []
    n_dim = len(attributes)
    for i in range(single_features.shape[0]):
        single_feature = single_features[i]
        if i < len(attributes): assert single_feature.sum() == 1
        description = get_concept_description(attributes, single_feature)
        if single_feature.sum() == 1:
            label = attributes_baselines[attributes.index(description)]
            single_feature_node = AOGNode(type="leaf", name=str(single_feature) + "(word)", label=label, layer_id=1, children=None, value=i)
            single_feature_nodes.append(single_feature_node)
        else:
            single_feature_node = AOGNode(type="AND", name=str(single_feature), label=description, layer_id=2, value=i)
            single_feature_node.extend_children([single_feature_nodes[j] for j in range(len(attributes)) if single_feature[j]])
            single_feature_nodes.append(single_feature_node)
    concept_nodes = []
    for i in range(concepts.shape[0]):
        concept = concepts[i]
        if not np.any(concept): continue
        concept_notation = generate_interaction_notation(np.arange(concept.shape[0])[concept].tolist(), n_dim=n_dim)
        concept_label = "{}\n{:+.2f}".format(concept_notation, eval_val[i])
        concept_node = AOGNode(type="AND", name=str(concept), label=concept_label, layer_id=3, value=eval_val[i])
        concept_node.extend_children([single_feature_nodes[j] for j in range(len(single_feature_nodes)) if concept[j]])
        concept_nodes.append(concept_node)
    root = AOGNode(type="+", name="+", label="+", layer_id=4, children=concept_nodes)
    aog = AOG(root=root, compulsory_nodes=single_feature_nodes[:len(attributes)])
    return aog