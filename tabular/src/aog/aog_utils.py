import numpy as np
from .graph import AOGNode, AOG
from pprint import pprint


def get_concept_description(attributes, single_feature, remove_linebreak_in_coalition=False):
    '''
    Generate a description of attributes
    :param attributes: basic attributes (in the original dataset) (n_dim,) list
    :param single_feature: Bool mask of a 'single features' (n_dim,)
    :return: a description string
    '''
    description = []
    n_linebreak = 0
    for i in range(single_feature.shape[0]):
        if single_feature[i]:
            n_linebreak += 1
            if remove_linebreak_in_coalition or n_linebreak >= 4:
                attribute = attributes[i].replace("\n", " ")
                attribute = attribute.replace("occupa- tion", "occupation") ############### ugly
                attribute = attribute.replace("relation- ship", "relationship") ############### ugly
                attribute = attribute.replace("educa- tion", "ecudation") ############### ugly
                attribute = attribute.replace("work- class", "workclass") ############### ugly
                description.append(attribute)
            else:
                n_linebreak += attributes[i].count("\n")
                description.append(attributes[i])
    description = "\n".join(description)
    return description


def replace_long_attributes(attributes):
    replace_table = {
        "shot length": "shot\nlength",
        "frame difference": "frame\ndifference",
        "fundamental frequency": "fundamental\nfrequency",
        "short time energy": "short time\nenergy",
        "spectral flux": "spectral\nflux",
        "zero crossing rate": "zero\ncrossing\nrate",
        "edge change ratio": "edge\nchange\nratio",
        "spectral roll off": "spectral\nroll off",
        "spectral centroid": "spectral\ncentroid",
        "hours per week": "hours per\nweek",
        "relationship": "relation-\nship",
        "occupation": "occupa-\ntion",
        "marital status": "marital\nstatus",
        "capital loss": "capital\nloss",
        "capital gain": "capital\ngain",
        # "education-num": "education",
        "education-num": "educa-\ntion",
        "workclass": "work-\nclass",
    }
    attributes_ = []
    for attribute in attributes:
        replaced = False
        for k, v in replace_table.items():
            if k in attribute:
                attributes_.append(attribute.replace(k, v))
                replaced = True
                break
        if not replaced:
            attributes_.append(attribute)
    return attributes_


def construct_AOG_v1(attributes, attributes_baselines, single_features, concepts, eval_val,
                     remove_linebreak_in_coalition=False):
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
    for i in range(single_features.shape[0]):
        single_feature = single_features[i]
        if i < len(attributes): assert single_feature.sum() == 1
        if single_feature.sum() == 1:
            description = get_concept_description(attributes, single_feature)
            label = attributes_baselines[attributes.index(description)]
            single_feature_node = AOGNode(type="leaf", name=description, label=label, layer_id=1, children=None)
            single_feature_nodes.append(single_feature_node)
        else:
            description = get_concept_description(attributes, single_feature,
                                                  remove_linebreak_in_coalition=remove_linebreak_in_coalition)
            single_feature_node = AOGNode(type="AND", name=description, label=description, layer_id=2)
            single_feature_node.extend_children([single_feature_nodes[j] for j in range(len(attributes)) if single_feature[j]])
            single_feature_nodes.append(single_feature_node)
    concept_nodes = []
    for i in range(concepts.shape[0]):
        concept = concepts[i]
        if not np.any(concept): continue
        concept_node = AOGNode(type="AND", name=str(concept), label="{:.2f}".format(eval_val[i]), layer_id=3, value=eval_val[i])
        concept_node.extend_children([single_feature_nodes[j] for j in range(len(single_feature_nodes)) if concept[j]])
        concept_nodes.append(concept_node)
    root = AOGNode(type="+", name="+", label="+", layer_id=4, children=concept_nodes)
    aog = AOG(root=root)
    return aog