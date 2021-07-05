import torch
from omegaconf import DictConfig
import torch.distributed as dist

from .base import BaseModel
from loss.soft_xent_loss import SoftXentLoss
from ..models.simsiam_arm import SiameseArm


class SwAVModel(BaseModel):
    def __init__(
        self,
        basic: DictConfig,
        backbone: DictConfig,
        mlp: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.online_network = self._prepare_model()
        self.prototypes = torch.nn.Linear(
            self.hparams.mlp.out_dim,
            self.hparams.basic.prototype_dim,
            bias=False,
        )
        self.criterion = SoftXentLoss(
            student_temp=self.hparams.basic.temperature,
            teacher_temp=1.0,
            teacher_softmax=False,
        )
        self.num_crops = basic.num_local_crops + basic.num_global_crops
        self.outputs = None

        self.is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if self.is_dist else 1
        if basic.queue_length % (basic.device_batch_size * world_size) != 0:
            raise ValueError("[Error] The length of queue "
                             "should be divisable by batch size!")

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def on_train_epoch_start(self):
        queue_length = self.hparams.basic.queue_length
        if queue_length > 0:
            # Create queue if epoch is enough
            start_epoch = self.hparams.basic.queue_start_epoch
            if (
                self.trainer.current_epoch >= start_epoch
                and not hasattr(self, "queue")
               ):
                self.register_buffer(
                    name="queue",
                    tensor=torch.zeros(
                        (self.hparams.basic.num_global_crops,
                         queue_length // dist.get_world_size(),
                         self.hparams.mlp.out_dim),
                        dtype=self.prototypes.weight.dtype,
                        device=self.prototypes.weight.device,
                    ),
                )
        return

    def on_after_backward(self):
        if self.current_epoch < self.hparams.basic.freeze_prototype_epoch:
            self.prototypes.weight.grad = None
        return

    # TODO: Check if changing to torch.nn.utils.weight_norm works
    @torch.no_grad()
    def _weight_normalize(self):
        w = self.prototypes.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)
        return

    def _multi_crop_forward(self, images: list) -> torch.Tensor:
        # Global features
        global_features = self.online_network(
            torch.cat(images[:self.hparams.basic.num_global_crops])
        )
        # Local features if have local crops
        if self.hparams.basic.num_local_crops > 0:
            local_features = self.online_network(
                torch.cat(images[self.hparams.basic.num_global_crops:])
            )
            whole_features = torch.cat((global_features, local_features))
        else:
            whole_features = global_features

        self.outputs = whole_features[:self.hparams.basic.device_batch_size]
        return torch.nn.functional.normalize(whole_features, dim=1, p=2)

    @torch.no_grad()
    def _queue_update_and_forward(
        self,
        index: int,
        features: torch.Tensor,
        online_score: torch.Tensor,
    ) -> torch.Tensor:

        bs = self.hparams.basic.device_batch_size
        # forward if queue is all filled
        if torch.all(self.queue[index, -1, :] != 0):
            queue_scores = self.prototypes(self.queue[index])
            online_score = torch.cat((queue_scores, online_score))
        # Pop the oldest batch and update
        self.queue[index, bs:] = self.queue[index, :-bs].clone()
        self.queue[index, :bs] = features[bs*index: bs*(index+1)]
        return online_score

    def _cal_total_loss(self, features: torch.Tensor, scores: torch.Tensor):
        bs = self.hparams.basic.device_batch_size
        loss = 0.0
        # Only use global crops to update queue and calculate assignments
        for idx in range(self.hparams.basic.num_global_crops):
            assignments = scores[bs*idx: bs*(idx+1)]
            # Use queue when its length > 0 and current epochs > start
            if hasattr(self, "queue"):
                assignments = \
                    self._queue_update_and_forward(idx, features, assignments)
            assignments = self._sinkhorn_knopp_algo(assignments, self.is_dist)
            # Use only the online assignments
            assignments = assignments[-bs:]

            sub_loss = 0.0
            for v in range(self.num_crops):
                if v == idx:
                    continue
                else:
                    student_output = scores[bs*v: bs*(v+1)]
                    sub_loss += self.criterion(student_output, assignments)
            loss += sub_loss / (self.num_crops - 1)
        return loss / self.hparams.basic.num_global_crops

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        images, _ = batch
        # Except the last one(weak augmentation)
        images = images[:-1]

        self._weight_normalize()
        features = self._multi_crop_forward(images)
        scores = self.prototypes(features)
        # TODO: the last batch is incompatible with the queue(trainer)
        loss = self._cal_total_loss(features, scores)

        self.log(
            "repr_loss", loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss

    def _prepare_model(self):
        online_network = SiameseArm(
            backbone=self.hparams.backbone.backbone,
            first_conv=self.hparams.backbone.first_conv,
            maxpool1=self.hparams.backbone.maxpool1,
            proj_hidden_dim=self.hparams.mlp.proj_hidden_dim,
            pred_hidden_dim=self.hparams.mlp.pred_hidden_dim,
            out_dim=self.hparams.mlp.out_dim,
            num_proj_mlp_layer=self.hparams.mlp.num_proj_mlp_layer,
            proj_last_bn=self.hparams.mlp.proj_last_bn,
            using_predictor=self.hparams.mlp.using_predictor,
        )
        return online_network

    @torch.no_grad()
    def _sinkhorn_knopp_algo(self, out: torch.Tensor, is_dist: bool):
        # Transpose here to be consistent with notion in original paper
        Q = torch.exp(out / self.hparams.basic.sinkhorn_epsilon).t()
        # Number of prototypes
        K = Q.shape[0]
        # Number of total samples
        B = Q.shape[1] * (dist.get_world_size() if is_dist else 1)

        # Make the matrix sum to 1
        sum_Q = torch.sum(Q)
        if is_dist:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for i in range(self.hparams.basic.sinkhorn_iter):
            # Normalize each row: total weight per prototype should be 1/K
            samples_sum = torch.sum(Q, dim=1, keepdim=True)
            if is_dist:
                dist.all_reduce(samples_sum)
            Q /= (samples_sum * K)

            # Normalize each column: total weight per sample should be 1/B
            prototypes_sum = torch.sum(Q, dim=0, keepdim=True)
            Q /= (prototypes_sum * B)

        Q *= B
        return Q.t()
