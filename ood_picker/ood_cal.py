import tqdm
import torch


@torch.no_grad()
def cal_ood_rate(
        out_domain_loader,
        threshold,
        model,
        temperature,
        perturbation,
):
    out_domain_scores = cal_softmax_scores(
        loader=out_domain_loader,
        model=model,
        temperature=temperature,
        perturbation=perturbation,
    )
    ood_sum = torch.sum(out_domain_scores <= threshold)
    ood_rate = ood_sum.cpu().numpy() / len(out_domain_scores)

    return ood_rate


@torch.no_grad()
def get_threshold(
        in_domain_loader,
        model,
        temperature,
        perturbation,
):
    in_domain_scores = cal_softmax_scores(
        loader=in_domain_loader,
        model=model,
        temperature=temperature,
        perturbation=perturbation,
    )
    sorted_scores, _ = torch.sort(in_domain_scores, descending=True)
    delta = sorted_scores[int(0.95*len(in_domain_scores))]
    return delta


@torch.no_grad()
def cal_softmax_scores(loader, model, temperature, perturbation):
    softmax_scores = list()
    pbar = tqdm.tqdm(loader)
    for data, _ in pbar:
        data = data.to("cuda")

        # Add perturbation for higher softmax scores
        if perturbation != 0:
            data = add_perturbation(
                data, model, temperature, alpha=perturbation)

        pred = model(data)
        pred /= temperature
        normed_pred = torch.softmax(pred, dim=1)
        score, _ = torch.max(normed_pred, dim=1)
        softmax_scores.append(score)
    return torch.cat(softmax_scores, dim=0)


@torch.enable_grad()
def add_perturbation(data, model, temperature, alpha=0.0011):
    data.requires_grad = True
    pred = model(data)
    psuedo_lbl = torch.softmax(pred.detach(), dim=1).argmax(dim=1)
    pred /= temperature
    loss = torch.nn.functional.cross_entropy(pred, psuedo_lbl)
    loss.backward()

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    std = torch.tensor([0.229, 0.224, 0.225], device=gradient.device)
    gradient /= std.reshape(1, 3, 1, 1)
    data = torch.add(data.data, -gradient, alpha=alpha)

    return data


@torch.no_grad()
def find_ood_files(
        loader,
        threshold,
        model,
        temperature,
        perturbation,
):
    ood_files = list()
    pbar = tqdm.tqdm(loader)
    for data, (_, file_name) in pbar:
        data = data.to("cuda")

        # Add perturbation for higher softmax scores
        if perturbation != 0:
            data = add_perturbation(
                data, model, temperature, alpha=perturbation)

        pred = model(data)
        pred /= temperature
        normed_pred = torch.softmax(pred, dim=1)
        score, _ = torch.max(normed_pred, dim=1)
        ood_idx = (score <= threshold).nonzero(as_tuple=True)[0].cpu()
        ood_files.extend([file_name[idx] for idx in ood_idx])
    print(f"[INFO] ood rate is: {len(ood_files)/len(loader.dataset):.2%}")
    return ood_files
