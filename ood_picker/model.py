import torch


def load_state_dict(path: str, model: torch.nn.Module):
    state_dict = torch.load(path)
    # Clean up weight
    pretrained_keys = list(state_dict["state_dict"].keys())
    for k in pretrained_keys:
        if "conv1.weight" not in k:
            _ = state_dict["state_dict"].pop(k)
        else:
            break
    dict_zip = zip(
        state_dict["state_dict"].items(),
        model.state_dict().items()
    )
    match_dict = {}
    for (s_k, s_v), (m_k, m_v) in dict_zip:
        if (m_k in s_k) and (s_v.shape == m_v.shape):
            match_dict[m_k] = s_v
    msg = model.load_state_dict(match_dict, strict=False)
    print(f"[ INFO ] Missing keys: {msg.missing_keys}")
    return model
