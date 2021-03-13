import logging

import torch


class NT_XentLoss(torch.nn.Module):
    def __init__(self, args):
        super(NT_XentLoss, self).__init__()
        self.tau = args.tau
        self.similarity_fn = torch.nn.CosineSimilarity(dim=1)
        logging.info('NT-XEnt Loss initialized. Tau = {:.2f}'.format(self.tau))

    def forward(self, z):
        """
        N: batch size
        Z: projection dim
        Input has shape (2N, Z)
        """
        # L2 normalization for each row
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        # (2N, Z) to (2N, Z, 1)
        z = z.unsqueeze(-1)
        # 2N x 2N matrix
        mat = torch.exp(self.similarity_fn(z, z.T) / self.tau)

        # take values 1 step above the diagonal,
        # but keep only alternating ones => these are for (i, j) pair
        # then repeat values for (j, i) pair (Cosine Similarity is symmetric)
        numerator = mat.diag(1)[::2].repeat_interleave(2)

        # sum value per row, but subtract value when i == j
        denominator = mat.sum(-1) - mat.diag()
        return (-(numerator / denominator).log()).mean()
