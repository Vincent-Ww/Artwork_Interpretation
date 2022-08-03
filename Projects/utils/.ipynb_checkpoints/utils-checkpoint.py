import requests
import torch
import numpy as np

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                
                
def text_progress2(minibatch):
#         print(f"text_progress {0}", minibatch)
    batch_tokens = [batch['cap'] for batch in minibatch]
    return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
            "cap": batch_tokens}

def text_progress(minibatch, text_field):
#         print(f"text_progress {0}", minibatch)
    batch_tokens = [batch['cap'] for batch in minibatch]
#         print(f"text_progress {1}", batch_tokens)
    padded_tokens = text_field.pad(batch_tokens)
#         print(f"text_progress {2}", padded_tokens)
    token_ids = text_field.numericalize(padded_tokens)
#         print(f"text_progress {3}", token_ids)
    return {"roi_feat": torch.from_numpy(np.array([batch['roi_feat'] for batch in minibatch])),
            "cap": token_ids}
