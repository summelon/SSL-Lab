# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
# from lightly.loss.memory_bank import MemoryBankModule
from .memory_bank import MemoryBankModule


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.
    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.
    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.
    Examples:
        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
    ):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.
        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.
            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)
            Returns:
                Contrastive Cross Entropy Loss value.
        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = \
            super(NTXentLoss, self).forward(out1, update=out0.requires_grad)

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        # MOCO
        if negatives is not None:
            # use negatives from memory bank
            # Be stored as parameters
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum('nc,ck->nk', out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
        # SimCLR
        else:
            if torch.distributed.is_available() \
                    and torch.distributed.is_initialized():
                out0_dist = SyncFunction.apply(out0)
                out1_dist = SyncFunction.apply(out1)
            else:
                out0_dist, out1_dist = out0, out1
            # use other samples from batch as negatives
            output = torch.cat((out0, out1), axis=0)
            output_dist = torch.cat((out0_dist, out1_dist), axis=0)

            # the logits are the similarity matrix divided by the temperature
            logits = torch.einsum('nc,mc->nm', output, output_dist)
            logits /= self.temperature
            # We need to removed the similarities of samples to themselves
            mask, labels = self._get_mask_and_label(batch_size, device)
            logits = logits[~mask].view(2*batch_size, -1)

        loss = self.cross_entropy(logits, labels)

        return loss

    def _get_mask_and_label(self, batch_size, device):
        """
            n: size of (out0, out1) of current world
            m: size of (out0_dist, out1_dist)
            x0, y0: mask index of out0
            x1, y1: mask index of out1
        """
        if torch.distributed.is_available() \
                and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            size = torch.distributed.get_world_size()
        else:
            rank, size = 0, 1
        n = 2 * batch_size
        m = 2 * batch_size * size
        x0 = torch.arange(batch_size, device=device)
        x1 = x0 + batch_size
        y0 = batch_size * rank + x0
        y1 = batch_size * (size + rank) + x0
        mask = torch.zeros(n, m)
        mask[torch.cat((x0, x1)), torch.cat((y0, y1))] = 1
        # Sub 1 since removed by mask
        labels = torch.cat((y1-1, y0))
        return mask.bool(), labels


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor)
                           for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
