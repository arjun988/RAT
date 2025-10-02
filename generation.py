import torch
import torch.nn.functional as F

@torch.no_grad()
def gpt_generate(model, tokenizer, prompt, max_len=100, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generated = input_ids
    past_kvs = None
    for _ in range(max_len):
        logits, past_kvs = model(generated, past_kvs=past_kvs, use_cache=True)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float("inf")
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0], skip_special_tokens=True)
