# Kaggle GPU script: restyle tinynagasaki render tiles with FLUX.1 Kontext dev + trained style LoRA.
import os, sys, subprocess, glob, json, time
subprocess.run('pip install -q -U "diffusers>=0.35" gguf peft "transformers>=4.51" accelerate sentencepiece protobuf', shell=True, check=True)
import torch
from huggingface_hub import hf_hub_download

CFG = json.load(open(glob.glob('/kaggle/input/**/job.json', recursive=True)[0]))
REPO = 'unsloth/FLUX.1-Kontext-dev'          # ungated mirror of black-forest-labs/FLUX.1-Kontext-dev
GGUF = ('QuantStack/FLUX.1-Kontext-dev-GGUF', CFG.get('gguf', 'flux1-kontext-dev-Q5_K_M.gguf'))
OUT = '/kaggle/working/out'; os.makedirs(OUT, exist_ok=True)
IN = os.path.dirname(glob.glob('/kaggle/input/**/job.json', recursive=True)[0])
tiles = sorted(glob.glob(f'{IN}/tiles/*.png'))
if CFG.get('only'): tiles = [t for t in tiles if os.path.basename(t)[:-4] in CFG['only']]
print('tiles', len(tiles), 'gpus', torch.cuda.device_count(), flush=True)

# ---- 1. prompt embeddings once (T5 in bf16 on CPU avoids fp16 overflow) ----
EMB = '/kaggle/working/emb.pt'
if not os.path.exists(EMB):
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    with torch.no_grad():
        pe, ppe, _ = p.encode_prompt(prompt=CFG['prompt'], prompt_2=None, device='cpu', max_sequence_length=512)
    torch.save({'pe': pe.float(), 'ppe': ppe.float()}, EMB)
    del p
print('embeddings ready', flush=True)

lora_path = '/kaggle/working/lora.safetensors'
if not os.path.exists(lora_path):
    subprocess.run(['wget', '-q', '-O', lora_path, CFG['lora_url']], check=True)
gguf_path = hf_hub_download(*GGUF)


def worker(gpu, my_tiles):
    import torch
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    from PIL import Image
    dev = f'cuda:{gpu}'
    tr = FluxTransformer2DModel.from_single_file(gguf_path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                                                 config=REPO, subfolder='transformer', torch_dtype=torch.float16)
    pipe = FluxKontextPipeline.from_pretrained(REPO, transformer=tr, text_encoder=None, text_encoder_2=None,
                                               tokenizer=None, tokenizer_2=None, torch_dtype=torch.float16)
    pipe.load_lora_weights(lora_path, adapter_name='style')
    pipe.set_adapters(['style'], adapter_weights=[CFG.get('lora_scale', 1.0)])
    pipe.to(dev)
    e = torch.load(EMB)
    pe = e['pe'].to(dev, torch.float16); ppe = e['ppe'].to(dev, torch.float16)
    for t in my_tiles:
        name = os.path.basename(t); out = f'{OUT}/{name}'
        if os.path.exists(out): continue
        t0 = time.time()
        img = Image.open(t).convert('RGB')
        g = torch.Generator(dev).manual_seed(CFG.get('seed', 7))
        res = pipe(image=img, prompt_embeds=pe, pooled_prompt_embeds=ppe, guidance_scale=CFG.get('guidance', 2.5),
                   num_inference_steps=CFG.get('steps', 24), height=img.height, width=img.width, generator=g,
                   max_area=img.width * img.height, _auto_resize=False).images[0]
        res.save(out)
        print(f'gpu{gpu} {name} {time.time() - t0:.1f}s', flush=True)

if __name__ == '__main__':
    import multiprocessing as mp
    n = max(1, torch.cuda.device_count())
    mp.set_start_method('spawn')
    ps = [mp.Process(target=worker, args=(g, tiles[g::n])) for g in range(n)]
    for p in ps: p.start()
    for p in ps: p.join()
    print('DONE', len(glob.glob(f'{OUT}/*.png')), flush=True)
