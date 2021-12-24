import torch

class ComplexProjectionModule(torch.nn.Module):

    def __init__(self, img_embed_dim = 384, text_embed_dim = 384):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(text_embed_dim, img_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(img_embed_dim, img_embed_dim)
        )
        self.img_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(img_embed_dim),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(img_embed_dim, img_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(img_embed_dim, img_embed_dim),
        )

    def forward(self, x):
        x1 = self.img_features(x[0])
        x2 = self.bert_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score
