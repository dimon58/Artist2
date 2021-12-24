import argparse

import more_itertools
import torch
import transformers
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from rudalle.utils import seed_everything, torch_tensors_to_pil_list

from services.utils import get_progress_bar
from settings import ALLOWED_MEMORY, DEVICE, PRETRAINED_PATH, HAFT_PRECISION

if ALLOWED_MEMORY < 4.5:
    DEVICE = 'cpu'
    DALLE_BS = 1
elif ALLOWED_MEMORY < 5:
    DALLE_BS = 1
elif ALLOWED_MEMORY <= 10.0:
    DALLE_BS = 5
else:
    DALLE_BS = 5

pretrained_path = PRETRAINED_PATH / 'rudalle'

dalle = get_rudalle_model('Malevich', pretrained=True, fp16=HAFT_PRECISION, device=DEVICE, cache_dir=pretrained_path)
tokenizer = get_tokenizer(cache_dir=pretrained_path)
vae = get_vae(dwt=True, cache_dir=pretrained_path)


def print_stats():
    import multiprocessing
    import torch
    import subprocess
    from psutil import virtual_memory

    print('ruDALL-E batch size:', DALLE_BS)

    total_memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if total_memory < ALLOWED_MEMORY:
        raise MemoryError
    print('Total GPU RAM:', round(total_memory, 2), 'Gb')

    ram_gb = round(virtual_memory().total / 1024 ** 3, 1)
    print('CPU:', multiprocessing.cpu_count())
    print('RAM GB:', ram_gb)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    if torch.__version__ >= '1.8.0':
        k = ALLOWED_MEMORY / total_memory
        torch.cuda.set_per_process_memory_fraction(k, 0)
        print('Allowed GPU RAM:', round(ALLOWED_MEMORY, 2), 'Gb')
        print('GPU part', round(k, 4))

    subprocess.call('nvidia-smi')


def generate_codebooks(text, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                       seed=None, use_cache=True, chat_id=None):
    if seed:
        seed_everything(seed)

    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')
    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)

    codebooks = []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            has_cache = False

            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)

            progress_bar = get_progress_bar(range(out.shape[1], total_seq_length), f'Рисование {chunk_bs} шт', chat_id)

            for idx in progress_bar:
                idx -= text_seq_length

                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)

                else:
                    logits, has_cache = dalle(out, attention_mask, has_cache=has_cache, use_cache=use_cache,
                                              return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)

            codebooks.append(out[:, -image_seq_length:].cpu())
    return codebooks


def generate_encoded(text, top_k, top_p, images_num, chat_id=None):
    return generate_codebooks(text, top_k=top_k, images_num=images_num, top_p=top_p, bs=DALLE_BS, chat_id=chat_id)


def decode_codebooks(codebooks, chat_id=None):
    pil_images = []

    progress_bar = get_progress_bar(torch.cat(codebooks).cpu(), 'Декодирование', chat_id)

    for _codebooks in progress_bar:
        with torch.no_grad():
            images = vae.decode(_codebooks.unsqueeze(0))
            pil_images += torch_tensors_to_pil_list(images)
    return pil_images


def generate(text='Пингвины радуются', top_k=1024, top_p=0.99, images_num=1, chat_id=None):
    codebooks = generate_encoded(text, top_k, top_p, images_num, chat_id=chat_id)
    images = decode_codebooks(codebooks, chat_id=chat_id)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, default='Пингвины радуются', help='Input text')
    parser.add_argument('-o', '--output', type=str, default='output.jpg', help='Output filename')
    parser.add_argument('--top_k', type=int, default=1024)
    parser.add_argument('--top_p', type=float, default=0.99)

    args = parser.parse_args()

    image = generate(args.text, args.top_k, args.top_p, 1)[0]
    image.save(args.output)


if __name__ == '__main__':
    seed_everything(6955)
    main()
