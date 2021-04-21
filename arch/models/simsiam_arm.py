import torch
from torch import nn
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.semi_supervised import Identity


def _torchvision_ssl_encoder(
    name: str,
    first_conv: bool,
    maxpool1: bool,
    pretrained: bool = False,
    return_all_feature_maps: bool = False,
) -> nn.Module:
    pretrained_model = getattr(resnets, name)(
        maxpool1=maxpool1,
        first_conv=first_conv,
        pretrained=pretrained,
        return_all_feature_maps=return_all_feature_maps
    )
    pretrained_model.fc = Identity()

    return pretrained_model


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        last_bn: bool = False,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        if last_bn:
            l3 = nn.Sequential(
                nn.Linear(hidden_dim, output_dim, bias=True),
                nn.BatchNorm1d(output_dim),
            )
        else:
            l3 = nn.Linear(hidden_dim, output_dim, bias=True)

        if num_layers == 3:
            self.mlp = nn.Sequential(l1, l2, l3)
        elif num_layers == 2:
            self.mlp = nn.Sequential(l1, l3)
        else:
            raise NotImplementedError("Only 2/3 layers MLP are implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x


class SiameseArm(nn.Module):
    def __init__(
        self,
        backbone: str,
        first_conv: bool,
        maxpool1: bool,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        out_dim: int,
        num_proj_mlp_layer: int = 2,
        mlp_last_bn: bool = False,
        using_predictor: str = True,
    ) -> None:
        super().__init__()

        self.using_predictor = using_predictor
        self.num_features = self._check_num_features(backbone)
        # Encoder
        self.backbone = _torchvision_ssl_encoder(
            name=backbone,
            first_conv=first_conv,
            maxpool1=maxpool1,
        )
        # Projector
        self.projector = MLP(
            input_dim=self.num_features,
            hidden_dim=proj_hidden_dim,
            output_dim=out_dim,
            last_bn=mlp_last_bn,
            num_layers=num_proj_mlp_layer,
        )
        if self.using_predictor:
            # Predictor
            self.predictor = MLP(
                input_dim=out_dim,
                hidden_dim=pred_hidden_dim,
                output_dim=out_dim,
                last_bn=mlp_last_bn,
                num_layers=2,
            )
        return

    def _single_forward(
        self,
        x: torch.Tensor,
        return_features: bool,
    ):
        f = self.backbone(x)[0]
        z = self.projector(f)
        if self.using_predictor:
            p = self.predictor(z)
            ssl_features = (z, p)
        else:
            ssl_features = z

        if return_features:
            out = (ssl_features, f)
        else:
            out = ssl_features
        return out

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor = None,
        return_features: bool = False,
    ):
        out0 = self._single_forward(x0, return_features)
        if x1 is None:
            return out0
        out1 = self._single_forward(x1, return_features)
        return out0, out1

    def _check_num_features(self, backbone):
        if backbone == "resnet18":
            num_features = 512
        elif backbone == "resnet50":
            num_features = 2048
        else:
            raise NotImplementedError("[ Error ] Backbone is not implemented!")
        return num_features
