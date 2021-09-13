import torchvision

from ood_cal import get_threshold, cal_ood_rate, find_ood_files
from data import get_data_loader, get_food11_loader, get_imagewang_loader
from model import load_state_dict


def main(args):
    partial_imagewoof_path = "/home/project/ssl/dataset/imagewoof0.1/train/"
    # Imagewang out-of-domain set
    out_domain_set_path = "/home/project/ssl/dataset/imagewang/train/"
    food11_path = "/dev/shm/"

    # Model
    pretrained_str_list = args.pretrained.split('-')
    arch = [s for s in pretrained_str_list if "resnet" in s][-1]
    in_domain_set = pretrained_str_list[pretrained_str_list.index(arch)+1]
    print(f"[INFO] Model arch: {arch}")
    print(f"[INFO] In-domain dataset: {in_domain_set}")
    model = getattr(torchvision.models, arch)(
        num_classes=11 if in_domain_set == "food11" else 10,
    )
    model = load_state_dict(
        path=args.pretrained,
        model=model,
    )
    model.to("cuda")
    model.eval()

    # In-domain dataset Loader
    if in_domain_set == "food11":
        in_domain_loader = get_food11_loader(
            path=food11_path,
            batch_size=args.batch_size,
            ratio=0.1,
        )
    elif in_domain_set == "imagewang":
        in_domain_loader = get_data_loader(
            path=partial_imagewoof_path,
            batch_size=args.batch_size,
        )
    else:
        raise NotImplementedError("[Error] In-domain dataset not implemented!")
    threshold = get_threshold(
        in_domain_loader=in_domain_loader,
        model=model,
        temperature=args.temperature,
        perturbation=args.perturbation,
    )
    print(f"[INFO] Threshold is: {threshold:.2%}")

    if args.dump_ood_file:
        assert in_domain_set == "imagewang", "[Error] In-domain dataset wrong!"
        # Imagewang out-of-domain set
        out_domain_loader = get_imagewang_loader(
            path=out_domain_set_path,
            batch_size=args.batch_size,
        )
        file_names = find_ood_files(
            loader=out_domain_loader,
            threshold=threshold,
            model=model,
            temperature=args.temperature,
            perturbation=args.perturbation,
        )
        with open(args.ood_file, 'w') as f:
            f.write('\n'.join(file_names))
    else:
        test_ood(
            threshold=threshold,
            model=model,
            temperature=args.temperature,
            perturbation=args.perturbation,
            batch_size=args.batch_size,
        )

    return


def test_ood(threshold, model, temperature, perturbation, batch_size):
    imagenette_path = "/dev/shm/imagenette/train"
    imagewoof_path = "/dev/shm/imagewoof/train"
    food11_path = "/dev/shm/"
    general_config = {
        "threshold": threshold,
        "model": model,
        "temperature": temperature,
        "perturbation": perturbation,
    }

    imagenette_loader = get_data_loader(
        path=imagenette_path,
        batch_size=batch_size,
    )
    imagewoof_loader = get_data_loader(
        path=imagewoof_path,
        batch_size=batch_size,
    )
    full_food11_loader = get_food11_loader(
        path=food11_path,
        batch_size=batch_size,
        ratio=1.0
    )
    ood_rate = cal_ood_rate(imagenette_loader, **general_config)
    print(f"[INFO] OOD rate on Imagenette is: {ood_rate:.2%}")
    ood_rate = cal_ood_rate(imagewoof_loader, **general_config)
    print(f"[INFO] OOD rate on Imagewoof is: {ood_rate:.2%}")
    ood_rate = cal_ood_rate(full_food11_loader, **general_config)
    print(f"[INFO] OOD rate on Food11 is: {ood_rate:.2%}")

    return


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained", type=str, required=True,
        help="Path to the pretrained weights"
    )
    parser.add_argument(
        "--perturbation", type=float, default=0.0011,
        help="Add perturbation when calculate softmax scores"
    )
    parser.add_argument(
        "--temperature", type=int, default=100,
        help="Temperature scaling for softmax"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--dump_ood_file", action="store_true",
        help="Dump ood image files name in Imagewang dataset"
    )
    parser.add_argument(
        "--ood_file", type=str, default="./ood_file.txt",
        help="The path to ood_file.txt"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
