import math
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from utils import d

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    """
    Self Attention layer.

    :param size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    """

    def __init__(self, emb_size: int = 128, heads: int = 6) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads

        self.tokeys = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.toqueries = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.tovalues = nn.Linear(emb_size, emb_size * heads, bias=False)
        self.output_layer = nn.Linear(emb_size * heads, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some
                  of the key, value vectors. [batch_size x 1 x key_len]

        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized
                representations of the queries.
        """
        b, t, e = x.size()
        h = self.heads
        assert (
            e == self.emb_size
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb_size})"

        keys = self.tokeys(x).view(b, t, h, e).transpose(1, 2)
        queries = self.toqueries(x).view(b, t, h, e).transpose(1, 2)
        values = self.tovalues(x).view(b, t, h, e).transpose(1, 2)

        # compute scaled dot-product self-attention
        queries = queries / math.sqrt(e)

        # for each word Wi the score with all other words Wj
        # for all heads inside the batch
        # [batch x num_heads x seq_len x seq_len]
        dot = torch.matmul(queries, keys.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, values)
        # Finally, we reshape back to [batch x seq_len x num_heads * embedding_size]
        context = context.transpose(1, 2).contiguous().view(b, t, h * e)
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x num_heads * embedding_size] -> [batch x seq_len x embedding_size]

        return self.output_layer(context)     


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
        Self attention -> Layer Norm -> Feed Forward -> Layer Norm

    :param emb_size: Size of the model embeddings.
    :param heads: Number of heads of the model.
    :param ff_hidden_mult: Int that will specify the size of the
        feed forward layer as a multiple of the embedding size.
    :param dropout: Dropout value to be applied between layers.
    """

    def __init__(
        self,
        emb_size: int = 128,
        heads: int = 6,
        ff_hidden_mult: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size),
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Encodes a sequence by passing it through 4 blocks:
            Self Attention -> Layer Norm -> Feed Forward -> Layer Norm

        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some
                  of the key, value vectors. [batch_size x 1 x key_len]
        """
        # Self Attention Block
        attended = self.attention(x, mask)

        # Normalization Block
        x = self.norm1(attended + x)
        x = self.do(x)

        # Feedforward Block
        fedforward = self.ff(x)

        # Normalization Block
        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x

class SimpleAttention(nn.Module):
    def __init__(self, emb_size: int = 128, key_hidden_size: int = 256, value_hidden_size: int = 256) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size

        self.tokeys = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.toqueries = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.tovalues = nn.Linear(emb_size, value_hidden_size, bias=False)
        # self.tokeys = nn.Sequential(
        #     nn.Linear(emb_size, key_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(key_hidden_size, key_hidden_size)
        # )
        # self.toqueries = nn.Sequential(
        #     nn.Linear(emb_size, key_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(key_hidden_size, key_hidden_size)
        # )
        # self.tovalues = nn.Sequential(
        #     nn.Linear(emb_size, value_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(value_hidden_size, value_hidden_size)
        # )
        # self.tovalues = nn.Identity()
        # self.output_layer = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some
                  of the key, value vectors. [batch_size x 1 x key_len]

        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized
                representations of the queries.
        """
        b, t, e = x.size()
        h = 1
        assert (
            e == self.emb_size
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb_size})"

        keys = self.tokeys(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
        queries = self.toqueries(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
        values = self.tovalues(x).view(b, t, h, self.value_hidden_size).transpose(1, 2)

        # compute scaled dot-product self-attention
        queries = queries / math.sqrt(self.key_hidden_size)

        # for each word Wi the score with all other words Wj
        # for all heads inside the batch
        # [batch x num_heads x seq_len x seq_len]
        dot = torch.matmul(queries, keys.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, values)
        # Finally, we reshape back to [batch x seq_len x num_heads * embedding_size]
        context = context.transpose(1, 2).contiguous().view(b, t, h * self.value_hidden_size)
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x num_heads * embedding_size] -> [batch x seq_len x embedding_size]

        return context
    
class PersonalizedSimpleAttention(nn.Module):
    "Personalized to Keys and to Queries"
    def __init__(self, emb_size: int = 128, key_hidden_size: int = 256, value_hidden_size: int = 256, n_persons: int = 32, personalization_type='prefix') -> None:
        super().__init__()
        self.n_person = n_persons
        self.emb_size = emb_size
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size

        self.tokeys = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.toqueries = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.tovalues = nn.Linear(emb_size, value_hidden_size, bias=False)

        self.personalization_type = personalization_type
        if(personalization_type == 'prefix' or personalization_type == 'add' or personalization_type == 'concat'):
            # use torch.Embedding for the personalized keys
            self.personalized_keys = nn.Embedding(n_persons, key_hidden_size)
            self.personalized_queries = nn.Embedding(n_persons, key_hidden_size)
        
        elif(personalization_type == 'linear'):
            # one Linear Layer per person
            self.personalized_keys = {}
            self.personalized_queries = {}
            
            self.personalized_keys = nn.ModuleDict(
                {str(i): nn.Linear(key_hidden_size, key_hidden_size) for i in range(n_persons)}
            )
            
            self.personalized_queries = nn.ModuleDict(
                {str(i): nn.Linear(key_hidden_size, key_hidden_size) for i in range(n_persons)}
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, person_idxs: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some
                  of the key, value vectors. [batch_size x 1 x key_len]
        :param person_idx: Index of the personalized key to use

        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized
                representations of the queries.
        """
        b, t, e = x.size()
        h = 1
        assert (
            e == self.emb_size
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb_size})"
        assert (    
            person_idxs.shape[0] == b
        ), f"Batch size of person_idxs ({person_idxs.shape[0]}) should match batch size of x ({b})"

        if(self.personalization_type == 'prefix' or self.personalization_type == 'add' or self.personalization_type == 'concat'):
            # Get personalized key and query embeddings
            personalized_key_emb = self.personalized_keys(person_idxs)  # [batch_size x person_rank]
            personalized_query_emb = self.personalized_queries(person_idxs)  # [batch_size x person_rank]
            personalized_key_emb = personalized_key_emb.unsqueeze(1).unsqueeze(2)  # [batch_size x 1 x 1 x person_rank]
            personalized_query_emb = personalized_query_emb.unsqueeze(1).unsqueeze(2)  # [batch_size x 1 x 1 x person_rank]
            
            if(self.personalization_type == 'prefix'):
                # prepend personal embeddings to x
                x = torch.cat([personalized_key_emb.squeeze(1), x], dim=1)
                # print(mask.shape, torch.ones((b, 1, 1), device=d()).bool().shape)
                mask = torch.cat([torch.ones((b, 1, 1), device=d()).bool(), mask], dim=2)
                t += 1 # increment sequence length

            # Compute keys, queries, and values
            keys = self.tokeys(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
            queries = self.toqueries(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
            values = self.tovalues(x).view(b, t, h, self.value_hidden_size).transpose(1, 2)

            if(self.personalization_type == 'add'):
                # add personal embeddings to k, q
                keys = keys + personalized_key_emb.expand(-1, -1, t, -1)
                queries = queries + personalized_query_emb.expand(-1, -1, t, -1)

            if(self.personalization_type == 'concat'):
                # concatenate personal embeddings to x
                raise NotImplementedError
            
        elif (self.personalization_type == 'linear'):
            keys = []
            queries = []
            values = self.tovalues(x).view(b, t, h, self.value_hidden_size).transpose(1, 2)
            for i in range(b):
                key = self.personalized_keys[str(person_idxs[i].item())](self.tokeys(x[i]))
                query = self.personalized_queries[str(person_idxs[i].item())](self.toqueries(x[i]))
                keys.append(key)
                queries.append(query)
            keys = torch.stack(keys).view(b, t, h, self.value_hidden_size).transpose(1, 2) 
            queries = torch.stack(queries).view(b, t, h, self.value_hidden_size).transpose(1, 2) 

        # compute scaled dot-product self-attention
        queries = queries / math.sqrt(self.key_hidden_size)

        # for each word Wi the score with all other words Wj
        # for all heads inside the batch
        # [batch x num_heads x seq_len x seq_len]
        dot = torch.matmul(queries, keys.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, values)
        # Finally, we reshape back to [batch x seq_len x num_heads * embedding_size]
        context = context.transpose(1, 2).contiguous().view(b, t, h * self.value_hidden_size)
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x num_heads * embedding_size] -> [batch x seq_len x embedding_size]

        return context


class ShallowTransformer(nn.Module):
    "Simple Attention Layer"
    def __init__(self, emb_size: int = 128, seq_length: int = 256, num_classes: int = 2, dropout=0.1, key_hidden_size=None, value_hidden_size=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        key_hidden_size = key_hidden_size
        value_hidden_size = value_hidden_size
        if(key_hidden_size is None):
            key_hidden_size = emb_size
        elif(value_hidden_size is None):
            value_hidden_size = emb_size

        # self.pos_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=seq_length)
        self.pos_embedding = PositionalEncoding(emb_size, dropout=dropout, max_len=seq_length)
        self.attention = SimpleAttention(emb_size=emb_size, key_hidden_size=key_hidden_size, value_hidden_size=value_hidden_size)
        self.toprobs = nn.Linear(value_hidden_size, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = x
        # b, t, e = tokens.size()
        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        positions = self.pos_embedding(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        x = tokens + positions
        x = self.do(x)
        x = self.attention(x, mask)
        x = self.do(x)
        x = x.mean(dim=1)
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)
    
class PersonalizedShallowTransformer(nn.Module):
    def __init__(self, emb_size: int = 128, seq_length: int = 256, num_classes: int = 2, dropout=0.1, key_hidden_size=None, value_hidden_size=None, n_persons=32, personalization_type=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        key_hidden_size = key_hidden_size
        value_hidden_size = value_hidden_size
        if(key_hidden_size is None):
            key_hidden_size = emb_size
        elif(value_hidden_size is None):
            value_hidden_size = emb_size

        self.pos_embedding = PositionalEncoding(emb_size, dropout=dropout, max_len=seq_length)
        self.attention = PersonalizedSimpleAttention(emb_size=emb_size, key_hidden_size=key_hidden_size, value_hidden_size=value_hidden_size, n_persons=n_persons, personalization_type=personalization_type)
        self.toprobs = nn.Linear(value_hidden_size, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, person_idxs: torch.Tensor) -> torch.Tensor:
        tokens = x
        # b, t, e = tokens.size()
        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        positions = self.pos_embedding(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        x = tokens + positions
        x = self.do(x)
        x = self.attention(x, mask, person_idxs)
        x = self.do(x)
        x = x.mean(dim=1)
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(
        self,
        num_classes: int,
        emb_size: int = 128,
        heads: int = 4,
        depth: int = 4,
        seq_length: int = 256,
        max_pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        :param num_classes: Number of classes.
        :param emb_size: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        :param dropout: dropout value to be applied between layers.
        """
        super().__init__()
        self.max_pool = max_pool
        # NOTE: no token embedding in this model as we take pre-embedded inputs
        # self.token_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)
        # self.pos_embedding = nn.Embedding(
        #     embedding_dim=emb_size, num_embeddings=seq_length
        # )
        self.pos_embedding = PositionalEncoding(emb_size, dropout=dropout, max_len=seq_length)

        self.tblocks = nn.ModuleList(
            [
                TransformerBlock(emb_size=emb_size, heads=heads, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.toprobs = nn.Linear(emb_size, num_classes)
        self.do = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Function that encodes the source sequence.
        :param x: Our vectorized source sequence embedding. [Batch_size x seq_len x emb_size]
        :param mask: Mask to be passed to the SelfAttention Block when encoding
                the x sequence -> check SelfAttention.

        :returns:
            -  predicted log-probability vectors for each token based on the preceding tokens.
                 [Batch_size x seq_len x n_classes]
        """
        tokens = x
        b, t, e = tokens.size()

        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(
        #     b, t, e
        # )
        positions = self.pos_embedding(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        x = tokens + positions
        x = self.do(x)

        for tblock in self.tblocks:
            x = tblock(x, mask)

        mask = mask.squeeze(1).float()
        expanded_mask = torch.repeat_interleave(mask, e, 1).view(b, t, e)
        x = torch.mul(expanded_mask, x)

        x = (
            x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        )  # pool over the time dimension
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)


class CTextTransformer(CTransformer):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_size: int = 128,
        heads: int = 4,
        depth: int = 4,
        seq_length: int = 256,
        max_pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        
        super().__init__(
            num_classes, emb_size, heads, depth, seq_length, max_pool, dropout
        )
        self.token_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tok)
        return super().forward(x, mask)


class Classifier(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 0.001) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = model.num_classes
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self._forward_step(batch, batch_idx)

        # log
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, _ = self._forward_step(batch, batch_idx)

        # log
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, __, y = batch
        loss, acc, preds = self._forward_step(batch, batch_idx)

        # log
        self.log_dict({"test/loss": loss, "test/acc": acc})

        # cache the predictions for analysis
        self.test_step_outputs.append([preds, y])
        return loss

    def _forward_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self.model(x, mask)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=-1)

        if self.num_classes == 2:
            acc = accuracy(preds, y, "binary")
        else:
            acc = accuracy(preds, y, "multiclass", num_classes=self.num_classes)
        return loss, acc, preds

    def on_test_epoch_end(self) -> None:
        preds, y = zip(*self.test_step_outputs)
        preds = torch.cat(preds).cpu().numpy()
        y = torch.cat(y).cpu().numpy()
        self.test_step_outputs.clear()

        # save predictions
        save_dir = self.logger.save_dir
        file_path = os.path.join(save_dir, "predictions", "test_preds.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame({"preds": preds, "labels": y})
        df.to_pickle(file_path)
        print(f"Predictions saved to {file_path}")
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class PersonalizedClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 0.001) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = model.num_classes
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self._forward_step(batch, batch_idx)

        # log
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, _ = self._forward_step(batch, batch_idx)

        # log
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, __, y, _ = batch
        loss, acc, preds = self._forward_step(batch, batch_idx)

        # log
        self.log_dict({"test/loss": loss, "test/acc": acc})

        # cache the predictions for analysis
        self.test_step_outputs.append([preds, y])
        return loss

    def _forward_step(self, batch, batch_idx):
        x, mask, y, person_idxs = batch
        y_hat = self.model(x, mask, person_idxs)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=-1)

        if self.num_classes == 2:
            acc = accuracy(preds, y, "binary")
        else:
            acc = accuracy(preds, y, "multiclass", num_classes=self.num_classes)
        return loss, acc, preds

    def on_test_epoch_end(self) -> None:
        preds, y = zip(*self.test_step_outputs)
        preds = torch.cat(preds).cpu().numpy()
        y = torch.cat(y).cpu().numpy()
        self.test_step_outputs.clear()

        # save predictions
        save_dir = self.logger.save_dir
        file_path = os.path.join(save_dir, "predictions", "test_preds.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame({"preds": preds, "labels": y})
        df.to_pickle(file_path)
        print(f"Predictions saved to {file_path}")
        return
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    # model = PersonalizedSimpleAttention(emb_size=128, key_hidden_size=128, value_hidden_size=128, n_persons=32, person_rank=4)
    # print(model)
    # # test forward pass 
    # x = torch.rand(4, 16, 128)
    # mask = torch.ones(4, 16, 1).bool()
    # person_idxs = torch.randint(0, 32, (4,))
    # # print(model.personalized_keys(person_idxs).shape)
    # out = model(x, mask, person_idxs)
    # print(out.shape)

    # test the personalized simple attention model for all personalization types
    model = PersonalizedSimpleAttention(emb_size=128, key_hidden_size=128, value_hidden_size=128, n_persons=32, personalization_type='prefix')
    print('Prefix:')
    # test forward pass
    x = torch.rand(4, 16, 128)
    mask = torch.ones(4, 16, 1).bool()
    person_idxs = torch.randint(0, 32, (4,))
    out = model(x, mask, person_idxs)
    print(out.shape)

    model = PersonalizedSimpleAttention(emb_size=128, key_hidden_size=128, value_hidden_size=128, n_persons=32, personalization_type='add')
    print('Add:')
    # test forward pass
    x = torch.rand(4, 16, 128)
    mask = torch.ones(4, 16, 1).bool()
    person_idxs = torch.randint(0, 32, (4,))
    out = model(x, mask, person_idxs)
    print(out.shape)

    # model = PersonalizedSimpleAttention(emb_size=128, key_hidden_size=128, value_hidden_size=128, n_persons=32, personalization_type='concat')
    # print('Concat:')
    # # test forward pass
    # x = torch.rand(4, 16, 128)
    # mask = torch.ones(4, 16, 1).bool()
    # person_idxs = torch.randint(0, 32, (4,))
    # out = model(x, mask, person_idxs)
    # print(out.shape)

    model = PersonalizedSimpleAttention(emb_size=128, key_hidden_size=128, value_hidden_size=128, n_persons=32, personalization_type='linear')
    print('Linear:')
    # test forward pass
    x = torch.rand(4, 16, 128)
    mask = torch.ones(4, 16, 1).bool()
    person_idxs = torch.randint(0, 32, (4,))
    out = model(x, mask, person_idxs)
    print(out.shape)
