from .mnist import MNIST, SimpleMNIST
from .mnist_one_vs_rest import SimpleIsThree
from .celeba import CelebA_binary


def get_dataset(data_root, dataset_name, eval_mode=False):
    if dataset_name == "mnist":
        return MNIST(data_root, eval_mode)
    elif dataset_name == "simplemnist":
        return SimpleMNIST(data_root, eval_mode)
    elif dataset_name == "simpleisthree":
        return SimpleIsThree(data_root)
    elif dataset_name.startswith("celeba_"):
        target_attribute = "_".join(dataset_name.split("_")[1:])
        return CelebA_binary(data_root, target_attribute)
    else:
        raise NotImplementedError