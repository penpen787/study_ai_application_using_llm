import example_2_1 as example
import torch

embedding_dim = 16
embed_layer = torch.nn.Embedding(len(example.str2idx), embedding_dim)

input_embeddings = embed_layer(torch.tensor(example.input_ids)) # (5,16)
print("input 1", input_embeddings)
input_embeddings = input_embeddings.unsqueeze(0) # (1, 5, 16)
print("input 2", input_embeddings)
input_embeddings.shape
print("input embeddings shape", input_embeddings.shape)
