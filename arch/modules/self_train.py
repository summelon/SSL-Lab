import copy
import torch
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import resnets
from torchmetrics import Accuracy

from .base import BaseModel


class SelfTrainModel(BaseModel):
    def __init__(
        self,
        basic: str,
        backbone: DictConfig,
        mlp: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        state_dict = torch.load(self.hparams.basic.ckpt_path)
        self.teacher = self._prepare_teacher(state_dict)
        self.student = self._prepare_student()
        # Load weights to teacher after deepcopying from teacher to student
        self._load_state_dict_to_specific_part(self.teacher, state_dict)
        # Freeze teacher after (perhaps) copy from teacher to student
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.classify_criterion = torch.nn.CrossEntropyLoss()
        self.t_acc = Accuracy()
        self.s_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        student_features = self.student(x)[0]
        student_logits = self.student.fc(student_features)
        return student_logits

    def _kd_loss(self, t_logits, s_logits, temperature):
        t_probs = torch.nn.functional.softmax(
            t_logits / temperature,
            dim=-1,
        )
        loss = self._tf_softmax_cross_entropy_with_logits(
            t_probs,
            s_logits / temperature,
        )
        loss *= temperature ** 2
        return loss

    def _tf_softmax_cross_entropy_with_logits(self, targets, logits):
        # Ref:
        #   https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
        loss = - targets * torch.nn.functional.log_softmax(logits, dim=-1)
        avg_loss = torch.sum(loss, dim=-1).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (images, _, _), labels = batch
        teacher_features = self.teacher(images)[0]
        teacher_logits = self.teacher.fc(teacher_features)
        student_logits = self(images)

        classify_loss = self.classify_criterion(
            student_logits,
            labels
        )
        kd_loss = self._kd_loss(
            teacher_logits,
            student_logits,
            self.hparams.basic.temperature,
        )

        alpha = self.hparams.basic.alpha
        loss = (1 - alpha) * classify_loss + alpha * kd_loss

        metrics = {
            "kd_loss": kd_loss,
            "classify_loss": classify_loss,
            "total_loss": loss,
            "teacher_acc": self.t_acc(teacher_logits.argmax(dim=1), labels),
            "student_acc": self.s_acc(student_logits.argmax(dim=1), labels),
        }
        self.log_dict(metrics, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        (_, _, images), labels = batch
        logits = self(images)
        loss = self.classify_criterion(logits, labels)
        self.log(
            "test_test_acc",
            self.test_acc(logits.argmax(dim=1), labels),
            prog_bar=True,
        )
        return loss

    def _prepare_teacher(self, state_dict):
        backbone_dict = state_dict["hyper_parameters"]["backbone"]
        teacher_network = getattr(resnets, backbone_dict["backbone"])(
            maxpool1=backbone_dict["maxpool1"],
            first_conv=backbone_dict["first_conv"],
        )
        teacher_network.fc = torch.nn.Linear(
            teacher_network.fc.in_features,
            state_dict["hyper_parameters"]["basic"]["num_classes"],
        )
        return teacher_network

    def _prepare_student(self):
        if self.hparams.basic.student_size == "same":
            # Using the same linear head as teacher, remain the fc no change
            student_network = copy.deepcopy(self.teacher)
        else:
            student_network = getattr(resnets, self.hparams.backbone.backbone)(
                maxpool1=self.hparams.backbone.maxpool1,
                first_conv=self.hparams.backbone.first_conv,
            )
            # Original ResNet has 1000 classes linear head
            student_network.fc = torch.nn.Linear(
                student_network.fc.in_features,
                self.teacher.fc.out_features,
            )
        return student_network
