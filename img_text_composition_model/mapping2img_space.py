import torch

class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim =384):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        # print([x[i].shape for i in range(len(x))])
        theta_linear = self.mapping(x[0])
        return theta_linear


class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim =384):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 48, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(16)

    def forward(self, x):
        # print(x)
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)

        final_vec = concat_x.reshape((concat_x.shape[0], 768))
        theta_conv = self.mapping(final_vec)
        return theta_conv
