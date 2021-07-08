import torch


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
        # Teacher
        # Remind the clone problem here
        teacher_pred = teacher_pred.detach()
        t = teacher_pred / self.teacher_temp
        if self.teacher_softmax:
            t = torch.softmax(t, dim=1)
        # Student
        s = torch.log_softmax(student_pred/self.student_temp, dim=1)

        cross_entropy = -(t * s).sum(dim=1)
        return cross_entropy.mean()
