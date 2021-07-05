import torch


class SoftXentLoss(torch.nn.Module):
    def __init__(
        self,
        student_temp: float,
        teacher_temp: float = 1.0,
        teacher_softmax: bool = True
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_softmax = teacher_softmax

    def forward(self, student_output, teacher_output):
        p = teacher_output.detach() / self.teacher_temp
        if self.teacher_softmax:
            p = torch.softmax(p, dim=1)
        q = torch.log_softmax(student_output/self.student_temp, dim=1)
        return (-p * q).sum(dim=1).mean()
