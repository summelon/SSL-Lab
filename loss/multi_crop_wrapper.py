import torch


class MultiCropLossWrapper(torch.nn.Module):
    def __init__(
        self,
        num_crops: int,
        loss_obj: torch.nn.Module,
    ):
        super().__init__()
        self.num_crops = num_crops
        self.loss = loss_obj
        return

    def forward(self, student_preds, teacher_preds):
        teacher_preds_list = teacher_preds.chunk(2)
        student_preds_list = student_preds.chunk(self.num_crops)
        self._check_batch_size(student_preds_list, teacher_preds_list)

        loss = 0.0
        # Outer loop: global crops, fixed in 2
        for o_idx in range(2):
            sub_loss = 0.0
            # Inner loop: global + local
            for i_idx in range(self.num_crops):
                if o_idx == i_idx:
                    continue
                else:
                    sub_loss += self.loss(
                        student_pred=student_preds_list[i_idx],
                        teacher_pred=teacher_preds_list[o_idx],
                    )
            loss += sub_loss / (self.num_crops - 1)
        loss /= 2
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
