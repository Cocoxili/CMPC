import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_euclidean_distance(A, B, normalized=False):
    """
    Pairwise Euclidean distance between two batches A and B. The batch size of A and B may not same.
    (A_{ij} - B_{ij})^2 = A_{ij}^2 - 2*A_{ij}B_{ij} + B_{ij}^2
    Euclidean Distance between each element of A and B.
    :param A: N x D
    :param B: M x D
    :return: N x M
    """
    eps = 1e-6
    a2 = torch.sum(A ** 2, dim=1).unsqueeze(1)
    b2 = torch.sum(B ** 2, dim=1).unsqueeze(0)
    ab = torch.matmul(A, B.transpose(0, 1))
    dist2 = a2 - 2 * ab + b2
    if normalized:
        dist2 = 2 - 2 * ab
    # print(dist2)
    # dist = torch.sqrt(dist2 + eps)
    # print(dist)
    return dist2


def pairwise_cosine_distance(A, B):
    """
    Pairwise Cosine distance between two batches A and B. The batch size of A and B may not same.
    Cosine Distance between each element of A and B.
    :param A: N x D
    :param B: M x D
    :return: N x M
    """
    eps = 1e-6
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    AB = torch.matmul(A, B.transpose(0, 1))
    return AB


def batchwise_euclidean_distance(x, y):
    """
    The batch size of x and y should be the same.
    :param x: (bs,dim)
    :param y: (bs,dim)
    :return: (bs)
    """
    return ((x - y) ** 2).sum(-1)


def batchwise_cosine_distance(x, y):
    """
    :param x: (bs,dim)
    :param y: (bs,dim)
    :return: (bs)
    """
    return F.cosine_similarity(x, y, dim=1)


def topk_in_2d_tensor(a, k):
    """
    Get topk maximum values and indexes in a matrix
    :return: (topk values,  2d indexes) tuple
    """
    h, w = a.shape
    b = a.view(-1)
    values, indices = b.topk(k)
    indices_2d = torch.cat(((indices // w).unsqueeze(1), (indices % w).unsqueeze(1)), dim=1)
    return (values, indices_2d)


def groupby_mean(value: torch.Tensor, labels: torch.LongTensor) -> (torch.Tensor, torch.LongTensor):
    """https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335"""

    """Group-wise average for (sparse) grouped tensors

    Args:
        value (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)

    Returns: 
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)

    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2, 0.2, 0.2],    #-> group / class 3
                             [0.4, 0.4, 0.4],    #-> group / class 3
                             [0.0, 0.0, 0.0]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)

        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.3000, 0.3000, 0.3000]])

        >>> new_labels
        tensor([0, 1, 5])
    """
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels))).cuda()

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).cuda().scatter_add_(0, labels, value)
    result = result / labels_count.float().unsqueeze(1)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels
