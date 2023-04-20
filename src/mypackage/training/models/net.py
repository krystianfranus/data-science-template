import torch
from torch import nn


class MF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
    ):
        super().__init__()
        self.embed_user = nn.Embedding(n_users, embed_size, sparse=True)
        self.embed_item = nn.Embedding(n_items, embed_size, sparse=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        embed_users = self.embed_user(users)
        embed_items = self.embed_item(items)
        output = torch.mul(embed_users, embed_items).sum(dim=1)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        embed_size = n_factors * (2 ** (n_layers - 1))
        self.embed_user = nn.Embedding(n_users, embed_size, sparse=True)
        self.embed_item = nn.Embedding(n_items, embed_size, sparse=True)

        mlp_modules = []
        for i in range(n_layers):
            input_size = n_factors * (2 ** (n_layers - i))
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.output_layer = nn.Linear(n_factors, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1, nonlinearity="sigmoid")

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        embed_users = self.embed_user(users)
        embed_items = self.embed_item(items)
        embed_concat = torch.cat((embed_users, embed_items), -1)
        mlp_output = self.mlp_layers(embed_concat)
        output = self.output_layer(mlp_output)
        return output.view(-1)
