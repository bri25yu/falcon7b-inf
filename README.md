# falcon7b-inf
## Example setup on machines
```bash
git clone https://github.com/bri25yu/falcon7b-inf
cd falcon7b-inf

conda env create --file environment.yml
conda activate falcon7b-inf

deepspeed run.py
```

## Example setup on Colab
```bash
git clone https://github.com/bri25yu/falcon7b-inf
cd falcon7b-inf

pip -q -q -q install -r requirements.txt

deepspeed run.py
```
