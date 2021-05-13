import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where((logits < min_values), torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_sequence(model, length, start_token=None, context=None, batch_size=1, temperature=1.0, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    past = None
    output = context
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output