import days.w2d1.bert_tests as bert_tests
from einops import rearrange, reduce, repeat
import math
import re
import torch as t
from torch import einsum
from torch.nn import functional as F
from torch import nn


def raw_attention_pattern(
    token_activations,
    num_heads,
    project_query,
    project_key,
):
    """Compute the multi-head attention pattern for a batch of sequences.
       Each attention pattern is computed as Q_h K_h^T, where Q_h and K_h
       are the query and key for the h-th head.

    Input:
        token_activations - Tensor[batch_size, seq_length, hidden_size(768)]
                            output of the previous layer
        num_heads         - int
                            number of heads to use
        project_query     - nn.Module, (Tensor[..., 768]) -> Tensor[..., 768]
                            a linear layer to project the input to the query space
        project_key       - nn.Module, (Tensor[..., 768]) -> Tensor[..., 768]
                            a linear layer to project the input to the key space
    Output:
        Tensor[batch_size, head_num, key_token: seq_length, query_token: seq_length]
        The attention pattern for each head.
    
    Hint: You can use rearrange to compute Q and K in one line.
    """
    raise NotImplementedError

if __name__ == "__main__":
    bert_tests.test_attention_pattern_fn(raw_attention_pattern)
    

def bert_attention(
    token_activations,
    num_heads: int, 
    attention_pattern,
    project_value,
    project_output,
):
    """Compute BERT attention output, which looks like
       softmax(attention_pattern) * project_value(token_activations)
    
    Input:
        token_activations - Tensor[batch_size, seq_length, hidden_size(768)]
                            output of the previous layer
        num_heads         - int
                            number of heads to use
        attention_pattern - Tensor[batch_size, head_num, key_token: seq_length, query_token: seq_length]
                            attention pattern for each head
        project_value     - nn.Module, (Tensor[..., 768]) -> Tensor[..., 768]
                            a linear layer to project the input to the value space
        project_output    - nn.Module, (Tensor[..., 768]) -> Tensor[..., 768]
                            a linear layer to project the output to the hidden size
    Output:
        Tensor[batch_size, seq_length, hidden_size]
        The output of the attention layer.
    """
    raise NotImplementedError

if __name__ == "__main__":
    bert_tests.test_attention_fn(bert_attention)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.project_key = nn.Linear(hidden_size, hidden_size)
        self.project_value = nn.Linear(hidden_size, hidden_size)
        self.project_output = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):  # b n l
        """Implement the multi-headed self-attention layer by
              1. Computing the attention pattern;
              2. Computing the attention output.
        Input:
            input - Tensor[batch_size, seq_length, hidden_size]
                    output of the previous layer
        """
        raise NotImplementedError
    
if __name__ == "__main__":
    bert_tests.test_bert_attention(MultiHeadedSelfAttention)


def bert_mlp(token_activations,
             linear_1: nn.Module, linear_2: nn.Module
):
    """Implement the MLP layer in BERT, which is a two-layer feed-forward network
       with a GELU activation in between.

    Input:
        token_activations - Tensor[batch_size, seq_length, hidden_size(768)]
                            output of the previous layer
        linear_1          - nn.Module, (Tensor[..., 768]) -> Tensor[..., 3072]
                            the first linear layer
        linear_2          - nn.Module, (Tensor[..., 3072]) -> Tensor[..., 768]
                            the second linear layer
    """
    raise NotImplementedError

if __name__ == "__main__":
    bert_tests.test_bert_mlp(bert_mlp)


class BertMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        self.lin1 = nn.Linear(input_size, intermediate_size)
        self.lin2 = nn.Linear(intermediate_size, input_size)

    def forward(self, input):
        return bert_mlp(input, self.lin1, self.lin2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int):
        super().__init__()
        self.weight = nn.Parameter(t.ones(normalized_dim))
        self.bias = nn.Parameter(t.zeros(normalized_dim))

    def forward(self, input):
        """Nothing to implement here, but you should read the code to understand LayerNorm
           Some helpful references:
                https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
                https://arxiv.org/abs/1607.06450
        """
        raise NotImplementedError
        input_m0 = input - input.mean(dim=-1, keepdim=True).detach()
        input_m0v1 = input_m0 / input_m0.std(dim=-1, keepdim=True, unbiased=False).detach()
        return input_m0v1 * self.weight + self.bias

if __name__ == "__main__":
    bert_tests.test_layer_norm(LayerNorm)

    
class BertBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout: float):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """Implement the BERT block, which is a multi-headed self-attention layer
           followed by a two-layer feed-forward network with a GELU activation in between.
           Each sub-layer has a residual connection around it, and is followed by a LayerNorm
        
        Input:
            input - Tensor[batch_size, seq_length, hidden_size]
                    output of the previous layer
        """
        raise NotImplementedError

if __name__ == "__main__":
    bert_tests.test_bert_block(BertBlock)    


# import transformers
# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
# print(tokenizer(['Hello, I am a sentence.']))

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding_matrix = nn.Parameter(t.randn(vocab_size, embed_size))

    def forward(self, input):
        """Nothing to implement here, but you should read the code to understand Embedding
           Some helpful references:
                https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        """
        raise NotImplementedError
        return self.embedding_matrix[input]

if __name__ == "__main__":
    bert_tests.test_embedding(Embedding)


def bert_embedding(
    input_ids,
    token_type_ids,
    position_embedding,
    token_embedding,
    token_type_embedding,
    layer_norm,
    dropout
):
    """Implement the embedding layer in BERT, which is the sum of the token embedding,
       the token type embedding, and the position embedding.
       This is followed by a LayerNorm and a Dropout.
       The output of the embedding layer is then passed to the BERT blocks.
    
    Input:
        input_ids          - Tensor[batch_size, seq_length]
                             the token ids of the input
        token_type_ids     - Tensor[batch_size, seq_length]
                             the token type ids of the input
        position_embedding - nn.Embedding
                             the position embedding
        token_embedding    - nn.Embedding
                             the token embedding
        token_type_embedding - nn.Embedding
                             the token type embedding
        layer_norm         - nn.Module
                             the layer norm
        dropout            - nn.Module
                             the dropout
    Output:
        Tensor[batch_size, seq_length, hidden_size]
        The output of the embedding layer.
    """
    raise NotImplementedError

if __name__ == "__main__":
    bert_tests.test_bert_embedding_fn(bert_embedding)


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        return bert_embedding(
            input_ids, token_type_ids, self.pos_embedding, self.token_embedding,
            self.token_type_embedding, self.layer_norm, self.dropout)

if __name__ == "__main__":
    bert_tests.test_bert_embedding(BertEmbedding)


class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers):
        """Initilaize a BERT model, consisting of
                1. An embedding layer (BertEmbedding);
                2. A stack of BERT blocks (BertBlock);
                3. A linear layer to project the output to the hidden size;
                4. A layer norm.
                5. Unembed: a linear layer to project the output to the vocabulary size.
        
        Input:
            vocab_size              - int
                                      the size of the vocabulary
            hidden_size             - int
                                      the hidden size
            max_position_embeddings - int
                                      the maximum position embedding
            type_vocab_size         - int
                                      the size of the token type vocabulary
            dropout                 - float
                                      the dropout rate
            intermediate_size       - int
                                      the hidden size of the intermediate layer in the MLP
            num_heads               - int
                                      the number of heads in the multi-headed self-attention
            num_layers              - int
                                      the number of BERT blocks in the stack      
        """
        super().__init__()

        """Uncomment this block after filling in the appropriate parameters.
        self.embed = BertEmbedding(?)
        self.blocks = nn.Sequential(*[
            ?
            for _ in range(num_layers)
        ])
        self.lin = nn.Linear(?)
        self.layer_norm = nn.LayerNorm(?)
        self.unembed = nn.Linear(?)
        """
        raise NotImplementedError
        

    def forward(self, input_ids):
        """Implement the forward pass of the BERT model.
        Input:
            input_ids - Tensor[batch_size, seq_length]
                        the token ids of the input
        """
        raise NotImplementedError


class BertWithClassify(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers, num_classes):
        """Initilaize a BERT model, consisting of:
                1. An embedding layer (BertEmbedding);
                2. A stack of BERT blocks (BertBlock);
                3. A linear layer to project the output to the hidden size;
                4. A layer norm.
                5. Unembed: a linear layer to project the output to the vocabulary size.
                6. A linear layer to project the output to the number of classes.
                7. A dropout layer for the classification head.
        """
        super().__init__()
        """Uncomment this block after filling in the appropriate parameters.
        self.embed = BertEmbedding(?)
        self.blocks = nn.Sequential(*[
            ?
            for _ in range(num_layers)
        ])
        self.lin = nn.Linear(?)
        self.layer_norm = nn.LayerNorm(?)
        self.unembed = nn.Linear(?)
        self.classification_head = nn.Linear(?)
        self.classification_dropout = nn.Dropout(?)
        """
        raise NotImplementedError
        

    def forward(self, input_ids):
        """Compute the vocabulary logits and the classification head output.
        """
        raise NotImplementedError


def mapkey(key):
    key = re.sub('^embedding\.', 'embed.', key)
    key = re.sub('\.position_embedding\.', '.pos_embedding.', key)
    key = re.sub('^lm_head\.mlp\.', 'lin.', key)
    key = re.sub('^lm_head\.unembedding\.', 'unembed.', key)
    key = re.sub('^lm_head\.layer_norm\.', 'layer_norm.', key)
    key = re.sub('^transformer\.([0-9]+)\.layer_norm', 'blocks.\\1.layernorm1', key)
    key = re.sub('^transformer\.([0-9]+)\.attention\.pattern\.',
                 'blocks.\\1.attention.', key)
    key = re.sub('^transformer\.([0-9]+)\.residual\.layer_norm\.',
                 'blocks.\\1.layernorm2.', key)
    
    key = re.sub('^transformer\.', 'blocks.', key)
    key = re.sub('\.project_out\.', '.project_output.', key)
    key = re.sub('\.residual\.mlp', '.mlp.lin', key)
    return key

if __name__ == "__main__":
    bert_tests.test_bert(Bert)    
    bert_tests.test_bert_classification(BertWithClassify) 
    my_bert = Bert(
        vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
        type_vocab_size=2, dropout=0.1, intermediate_size=3072, 
        num_heads=12, num_layers=12
    )
    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                    if not k.startswith('classification_head')}
    my_bert.load_state_dict(mapped_params)
    bert_tests.test_same_output(my_bert, pretrained_bert)
