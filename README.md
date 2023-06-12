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

## Using ctranslate2
```bash
python ctranslate2_falcon.py --model tiiuae/falcon-7b-instruct --quantization float16 --output_dir falcon-7b-instruct
```

## Benchmarks
Output from `test_exact.py` on a Google Colab A100 GPU with 40GB of GPU VRAM
```
Init time
	Base model 13.370s
	Reimpl best match model 12.226s
	Reimpl model 12.226s
Inference time
	Base model 46.104s
	Reimpl best match model 42.374s
	Reimpl model 9.104s
Base and reimpl best match model outputs match
Base and reimpl model outputs DO NOT match!
Base model output
In his senior year, James was named the "Mr. Basketball" of Ohio and was named the "Mr. Basketball" of the state of Ohio. He was also named the "Mr. Basketball" of the United States.[14]: 28  He was the first player in the history of the Akron area to be named to the McDonald's All-American team.[14]: 29  He was also named to the "All-Ohio" team, the "Ohio Capital Conference" team, and the "Ohio North Coast Conference" team.[14]: 30  He was also named the "Ohio Capital Conference" Player of the Year.[14]: 31  He was also named the "Ohio Capital Conference" Player of the Year for the second time in his career.[14]: 32  He was also named the "Ohio Capital Conference" Player of the Year for the third time in his career.[14]: 33  He was also named the "Ohio Capital Conference" Player of the Year for the fourth time in his career.[14]: 34  He was also named the "Ohio Capital Conference" Player of the Year for the fifth time in his career.[14]: 35  He was also named the "Ohio Capital Conference" Player of the Year for the sixth time in his career.[14]:  

Reimpl model output
In his senior year, James was named the "Mr. Basketball" of Ohio and was named the "Mr. Basketball" of the state of Ohio. He was also named the "Mr. Basketball" of the United States.[14]: 28  He was the first player in the history of the Akron area to be named to the McDonald's All-American team.[14]: 29  He was also named to the "All-Ohio" team, the "Ohio Capital Conference" team, and the "Ohio North Coast Conference" team.[14]: 30  He was also named to the "Ohio Capital Conference" team for the second time in his career.[14]: 31  He was also named to the "Ohio North Coast Conference" team for the third time in his career.[14]: 32  He was also named to the "Ohio Capital Conference" team for the fourth time in his career.[14]: 33  He was also named to the "Ohio Capital Conference" team for the fifth time in his career.[14]: 34  He was also named to the "Ohio Capital Conference" team for the sixth time in his career.[14]: 35  He was also named to the "Ohio Capital Conference" team for the seventh time in his career.[14]: 36  He 
```

Output from `test_ctranslate2.py` on a Google Colab A100 GPU with 40GB of GPU VRAM:
```
Init time
	CTranslate2 model and generation 11.021s
	HF model and generation 15.132s
Inference time
	CTranslate2 model and generation 5.324s
	HF model and generation 8.633s
Ctranslate2 and HF outputs DO NOT match!
Ctranslate2 output

In his senior year, James was named the "Mr. Basketball" of Ohio and was named the "Mr. Basketball" of the state of Ohio. He was also named the "Mr. Basketball" of the United States.[14]: 28  He was also named the "Mr. Basketball" of the state of Ohio for the second time in his senior year.[14]: 29  He was also named the "Mr. Basketball" of the United States for the second time in his senior year.[14]: 30  He was also named the "Mr. Basketball" of the state of Ohio for the third time in his senior year.[14]: 31  He was also named the "Mr. Basketball" of the United States for the third time in his senior year.[14]: 32  He was also named the "Mr. Basketball" of the state of Ohio for the fourth time in his senior year.[14]: 33  He was also named the "Mr. Basketball" of the United States for the fourth time in his senior year.[14]: 34  He was also named the "Mr. Basketball" of the state of Ohio for the fifth time in his senior year.[14]: 35  He was also named the "Mr. Basketball" of the United States for the fifth time in his senior 

HF output
In his senior year, James was named the "Mr. Basketball" of Ohio and was named the "Mr. Basketball" of the state of Ohio. He was also named the "Mr. Basketball" of the United States.[14]: 28  He was the first player in the history of the Akron area to be named to the McDonald's All-American team.[14]: 29  He was also named to the "All-Ohio" team, the "Ohio Capital Conference" team, and the "Ohio North Coast Conference" team.[14]: 30  He was also named to the "Ohio Capital Conference" team for the second time in his career.[14]: 31  He was also named to the "Ohio North Coast Conference" team for the third time in his career.[14]: 32  He was also named to the "Ohio Capital Conference" team for the fourth time in his career.[14]: 33  He was also named to the "Ohio Capital Conference" team for the fifth time in his career.[14]: 34  He was also named to the "Ohio Capital Conference" team for the sixth time in his career.[14]: 35  He was also named to the "Ohio Capital Conference" team for the seventh time in his career.[14]: 36  He 
```
