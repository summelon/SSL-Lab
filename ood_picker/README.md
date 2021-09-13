# Out-of-Domain Data Picker

## Quick start
- Change the dataset path in `main.py`
- Use the supervised model pretrained on the fraction of labeled data
```shell=
python3 main.py \
    --dump_ood_file \
    `# The in-domain dataset will be the postfix in pretrained weight name, imagewang in this case` \
    --pretrained /path/to/your/weights/supervised-resnet50-imagewang-epoch=00001-step=9-supervised_val_acc=0.888.ckpt
```
- Use the `ood_file.txt` to filter out-of-domain data in Imagewang
