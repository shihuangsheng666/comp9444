import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import csv
import nltk
from transformers import AutoTokenizer
import sacrebleu
from rouge import Rouge
import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import string
import pandas as pd
import nltk
nltk.download('wordnet')
# def prepare_data(all_data, part, clean = True):
#     table = str.maketrans('','',string.punctuation)  #remove all punctuation 
#     captions = all_data['captions']
#     captions = pd.DataFrame(captions)
#     captions = captions.sort_values(by ='image_id')  
#     num_to_sample = len(captions['image_id'].unique())//part   
#     sampled_image_ids = captions['image_id'].drop_duplicates().sample(n=num_to_sample, random_state=11)
#     sampled_captions_df = captions[captions['image_id'].isin(sampled_image_ids)]
#     #print(sampled_captions_df)
#     clean_captions = sampled_captions_df.to_dict(orient='records')
#     #print(clean_captions[0])
#     if clean:
#       for j in range(len(clean_captions)):        
#           sentence = clean_captions[j]['caption'].replace("-"," ")
#           # remove punctuation
#           sentence = sentence.translate(table)
#           split_sentence = sentence.split()
#           # convert all to lower case
#           split_sentence = [wrd.lower() for wrd in split_sentence]
#           # remove 's and a that does not help with training
#           split_sentence = [wrd for wrd in split_sentence if(len(wrd)>1)]
#           # remove all string like numbers
#           split_sentence = [wrd for wrd in split_sentence if(wrd.isalpha())]
#           sentence = ' '.join(split_sentence)
#           clean_captions[j]['caption'] = sentence
#     #print(clean_captions[0])
#     correspond_embbeding = []
#     for i in range(len(clean_captions)):
#         correspond_embbeding.append(all_data['clip_embedding'][clean_captions[i]['clip_embedding']])
#         clean_captions[i]['clip_embedding'] = i
#     #clean_embbeding = {i: value for i, value in enumerate(correspond_embbeding)}  
    
#     all_data_clean = {'captions':clean_captions, 'clip_embedding':correspond_embbeding}
#     return all_data_clean
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        # clean_all_data = prepare_data(all_data, 10, False)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        
        captions_raw = all_data["captions"]
        
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        
class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)

        # Transform the prefix using the linear layer
        transformed_prefix = self.prefix_dim_reduction(prefix)

        prefix_projections = self.clip_project(transformed_prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # Add a linear layer for reducing the dimension of the prefix
        self.prefix_dim_reduction = nn.Linear(1024, 512)

        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)

# class ClipCaptionModel(nn.Module):

#     def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

#     def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None):
#         embedding_text = self.gpt.transformer.wte(tokens)
#                 # Debugging: Print the shape of the prefix and embedding_text
#         print(f"Shape of prefix: {prefix.shape}")
#         print(f"Shape of embedding_text: {embedding_text.shape}")
#         prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
#         #print(f"Shape of prefix_projections after view: {prefix_projections.shape}")
#         embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
#         if labels is not None:
#             dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
#             labels = torch.cat((dummy_token, tokens), dim=1)
#         out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
#         return out

#     def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
#                  num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
#         super(ClipCaptionModel, self).__init__()
#         self.prefix_length = prefix_length
#         self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
#         if mapping_type == MappingType.MLP:
#             self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
#                                      self.gpt_embedding_size * prefix_length))
#         else:
#             self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
#                                                                      clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

def compute_metrics(references, hypotheses, val_loss):
    
    # Tokenize the hypotheses and references
    hypotheses_tokens = [hyp.split() for hyp in hypotheses]
    references_tokens = [[ref.split() for ref in ref_group] for ref_group in references]
    
    # Convert references back to string format for rouge scoring
    references_str = [' '.join(ref_group[0]) for ref_group in references_tokens]
    
    # For BLEU
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references_tokens, hypotheses_tokens, smoothing_function=smoothie)
    
    # For ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references_str, avg=True)
    rouge_l = rouge_scores['rouge-l']['f']
    
    # For METEOR
    meteor_scores = [meteor_score.single_meteor_score(ref.split(), hyp.split()) for ref, hyp in zip(references_str, hypotheses)]
    meteor = sum(meteor_scores) / len(meteor_scores)
    
    # Compute Perplexity from Validation Loss
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    
    return bleu_score, rouge_l, val_loss, perplexity, meteor


def train(dataset, model, args, lr=2e-5, warmup_steps=5000, output_dir=".", output_prefix="", val_dataloader=None, tokenizer=None):
    # ... rest of your code ...


    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    # Initialize lists to hold the metric values across epochs
    bleu_scores, rouge_scores, val_losses, perplexities, meteor_scores = [], [], [], [], []

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()

        progress = tqdm(total=len(train_dataloader), desc=output_prefix)

        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)

            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )

        progress.close()

        #Begin validation evaluation
        model.eval()
        val_loss = 0
        references = []
        hypotheses = []

        for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(tokens, prefix, mask)

            logits = outputs.logits[:, dataset.prefix_length - 1: -1]

            val_loss += nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0).item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for i in range(preds.shape[0]):
                ref_caption = dataset.captions[idx * batch_size + i]
                hyp_caption = tokenizer.decode(preds[i], skip_special_tokens=True)
                references.append([ref_caption])
                hypotheses.append(hyp_caption)

        val_loss /= len(val_dataloader)
        bleu, rouge, v_loss, perplexity, meteor = compute_metrics(references, hypotheses, val_loss)

        print(f"Epoch {epoch}: BLEU: {bleu}, ROUGE: {rouge}, Val Loss: {v_loss}, Perplexity: {perplexity}, METEOR: {meteor}")

        # Append the metric values for this epoch to the lists
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        val_losses.append(v_loss)
        perplexities.append(perplexity)
        meteor_scores.append(meteor)

        model.train()

        # End validation evaluation
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

    # After the training loop, create plots for each metric
    epochs_range = range(epochs)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, bleu_scores, label='BLEU')
    plt.legend(loc='upper right')
    plt.title('BLEU Score')

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, rouge_scores, label='ROUGE')
    plt.legend(loc='upper right')
    plt.title('ROUGE Score')

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Validation Loss')

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, perplexities, label='Perplexity')
    plt.legend(loc='upper right')
    plt.title('Perplexity')

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, meteor_scores, label='METEOR')
    plt.legend(loc='upper right')
    plt.title('METEOR Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.show()

    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--val_data', default='./data/coco/oscar_split_ViT-B_32_val.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    val_data_file = args.val_data
    val_dataset = ClipCocoDataset(val_data_file, args.prefix_length, normalize_prefix=args.normalize_prefix)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, drop_last=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2') 
    train(dataset, model, args, lr=2e-5, warmup_steps=5000, output_dir=args.out_dir, output_prefix=args.prefix, val_dataloader=val_dataloader, tokenizer=tokenizer)




if __name__ == '__main__':
    main()