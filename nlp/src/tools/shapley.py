import numpy as np

def estimate_shapley(
        interactions: np.ndarray,
        masks: np.ndarray,
        n_patterns: int
):
    '''
    This function returns the estimated shapley value given the patterns and their interactions
    :param interactions:
    :param masks:
    :param n_patterns:
    :return:
    '''
    estimated_shapley = np.zeros(masks.shape[1])
    CI_order = np.argsort(-np.abs(interactions))

    # estimate the shapley value
    for i in range(n_patterns):
        mask = masks[CI_order][i]
        interaction = interactions[CI_order][i]
        if not np.any(mask): continue
        estimated_shapley[mask] += interaction / np.sum(mask)

    return estimated_shapley