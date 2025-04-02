import torch
from torch import nn


class MF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        user_history_based: bool,
    ):
        super().__init__()
        self.user_history_based = user_history_based

        if self.user_history_based:
            self.embed_item = nn.Embedding(n_items + 1, embed_size, sparse=True)
        else:
            self.embed_user = nn.Embedding(n_users, embed_size, sparse=True)
            self.embed_item = nn.Embedding(n_items, embed_size, sparse=True)

        self._init_weights()

    def _init_weights(self):
        if not self.user_history_based:
            nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        if self.user_history_based:
            embed_users = self.embed_item(users).mean(dim=1)
        else:
            embed_users = self.embed_user(users)
        embed_items = self.embed_item(items)
        output = torch.mul(embed_users, embed_items).sum(dim=1)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        n_layers: int,
        dropout: float,
        user_history_based: bool,
    ):
        super().__init__()
        self.user_history_based = user_history_based

        if self.user_history_based:
            self.embed_item = nn.Embedding(n_items + 1, embed_size, sparse=True)
        else:
            self.embed_user = nn.Embedding(n_users, embed_size, sparse=True)
            self.embed_item = nn.Embedding(n_items, embed_size, sparse=True)
        mlp_modules: list[nn.Module] = []
        for i in range(n_layers):
            input_size = (2 * embed_size) // (2**i)
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.output_layer = nn.Linear(input_size // 2, 1)

        self._init_weights()

    def _init_weights(self):
        if not self.user_history_based:
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
        if self.user_history_based:
            embed_users = self.embed_item(users).mean(dim=1)
        else:
            embed_users = self.embed_user(users)
        embed_items = self.embed_item(items)
        embed_concat = torch.cat((embed_users, embed_items), -1)
        mlp_output = self.mlp_layers(embed_concat)
        output = self.output_layer(mlp_output)
        return output.view(-1)
