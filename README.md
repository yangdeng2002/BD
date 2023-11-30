# Blurred-Dilated Method
This repository is the official Pytorch code  for our paper **Blurred-Dilated Method for Adversarial Attacks (NeurIPS'2023)**.



## Prepare

Please download our PyTorch pretrained models at [here](https://drive.google.com/file/d/10UChaDQ5PSGGNTxNg9HzrozTbsSsgjmh/view?usp=drive_link), and then put it into `./weight/`.

## Generate adversarial examples
  
  Using `attack.py` to implement our BD method with MI-FGSM as the optimization algorithm and ResNet-50 as the source model. You can run this attack as following:
  
```
python attack.py
```
  
  And adversarial examples will be generated in directory`./outputs`.

 ## Evaluations

Running verify.py to evaluate the attack success rate:

```
python verify.py
```

## Citation

If you find our work is useful, please consider citing our paper.
