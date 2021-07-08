import torch
import torch.distributed as dist


class SoftXentLoss(torch.nn.Module):
    def __init__(
        self,
        student_temp: float,
        teacher_temp: float = 1.0,
        teacher_softmax: bool = True,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_softmax = teacher_softmax

    def forward(self, student_pred, teacher_pred):
        # Teacher(full): softmax( (t_pred - center) / t_temp )
        # Remind the clone problem here
        teacher_pred = teacher_pred.detach()
        t = teacher_pred / self.teacher_temp
        if self.teacher_softmax:
            t = torch.softmax(t, dim=1)
        # Student
        s = torch.log_softmax(student_pred/self.student_temp, dim=1)

        cross_entropy = -(t * s).sum(dim=1)
        return cross_entropy.mean()


class MultiCropSoftXentLoss(SoftXentLoss):
    def __init__(
        self,
        num_crops: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        # DINO center
        k_dim: int = 65536,
        center_momentum: float = 1.0,
        # SimEstimator temperature annealing
        anneal_temp: bool = False,
    ):
        # Initialize
        super().__init__(
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            teacher_softmax=True,
        )
        self.num_crops = num_crops
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # DINO: Center will not update when momentum = 1, thus None
        if center_momentum != 1.0:
            self.register_buffer("center", torch.zeros(1, k_dim))
        # SimEstimator temperature annealing
        if anneal_temp:
            from math import ceil, log10
            self.scale_s = torch.nn.Parameter(torch.ones(1))
            self.lower_bound = 4 + 2 * ceil(log10(k_dim))
        return

    def forward(self, student_preds, teacher_preds):
        loss = 0.0
        # DINO center
        if hasattr(self, "center"):
            teacher_preds_clone = teacher_preds.clone()
            teacher_preds -= self.center
            self._update_center(teacher_preds_clone)
        # SimEstimator temperature annealing
        if hasattr(self, "scale_s"):
            loss += self._temp_assign_and_anneal()

        teacher_preds_list = teacher_preds.chunk(2)
        student_preds_list = student_preds.chunk(self.num_crops)
        self._check_batch_size(student_preds_list, teacher_preds_list)

        # Outer loop: global crops, fixed in 2
        for o_idx in range(2):
            sub_loss = 0.0
            # Inner loop: global + local
            for i_idx in range(self.num_crops):
                if o_idx == i_idx:
                    continue
                else:
                    sub_loss += super().forward(
                        student_pred=student_preds_list[i_idx],
                        teacher_pred=teacher_preds_list[o_idx],
                    )
            # Divided by 2(global crops)
            loss += 0.5 * (sub_loss / (self.num_crops - 1))
        return loss

    def _check_batch_size(self, student_preds: list, teacher_preds: list):
        def equal_check(preds):
            return all([p.shape[0] == preds[0].shape[0] for p in preds])

        if not equal_check(student_preds+teacher_preds):
            raise ValueError(
                "[Error] The batch size of each prediction should be the same!"
                f"\n\t The length of s_preds list: {len(student_preds)}"
                f", the length of t_preds list: {len(teacher_preds)}",
                f"\n\t Batch size of s_preds[-1]: {student_preds[-1].shape[0]}"
                f", batch size of t_preds[-1]: {teacher_preds[-1].shape[0]}"
            )
        return

    def _temp_assign_and_anneal(self):
        self.teacher_temp = self.student_temp = 1 / self.scale_s
        annealing_loss = 0.5 * (self.lower_bound - self.scale_s).pow(2)
        return annealing_loss

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """
        For DINO: Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = \
            batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )
        return
