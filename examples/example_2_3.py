import example_2_1 as example
import torch

embedding_dim = 16
max_position = 12
embed_layer = torch.nn.Embedding(len(example.str2idx), embedding_dim)
position_embed_layer = torch.nn.Embedding(max_position, embedding_dim)

position_ids = torch.arange(len(example.input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)
token_embeddings = embed_layer(torch.tensor(example.input_ids)) # (5,16)
token_embeddings = token_embeddings.unsqueeze(0) # (1,5,16)
input_embeddings = token_embeddings + position_encodings
print("shape", input_embeddings) # tensor ([1,5,26])
