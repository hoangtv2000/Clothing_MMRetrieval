import timm
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from sentence_transformers import SentenceTransformer
from img_model.vit_extractor import ViT

from .complex_projection_module import ComplexProjectionModule
from .mapping2img_space import LinearMapping, ConvMapping
from triplet_loss.triplet_loss import SoftTripletLoss

from .util_layers import  NormalizationLayer


class ComposeTransformers(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_embed_dim = 384
        self.text_embed_dim = 384
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and config.cuda) else "cpu")

        self.triplet_loss = SoftTripletLoss()

        self.img_model = ViT(self.config)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v1')

        self.norm_scale = NormalizationLayer()

        """
        """
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0]))

        """
        """
        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )

        """
        """
        # self.encoderWithConv = torch.nn.Sequential(
        #     ComplexProjectionModule(),
        #     ConvMapping()
        # )

        """
        """
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.img_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.img_embed_dim, self.img_embed_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.img_embed_dim, self.img_embed_dim)
        )

        """
        """
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.img_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.img_embed_dim, self.text_embed_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.text_embed_dim, self.text_embed_dim)
        )


        """
        """
        self.normalization_layer = NormalizationLayer()



    def extract_text_feature(self, text_query):
        """Sentence embedding by text model.
        """
        return torch.tensor(self.text_model.encode(text_query)).to(self.device)


    def extract_img_feature(self, img):
        """Extract img feature by backbone.
        """
        return self.img_model(img).to(self.device)


    def compose_img_text_features(self, img_features, text_features, \
                    CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False)):
        """
        """
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        # theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))
        # theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        theta = theta_linear * self.a[1]

        dct_with_representations = {"repres": theta,
                                    "repr_to_compare_with_source": self.decoder(theta),
                                    "repr_to_compare_with_mods": self.txtdecoder(theta),
                                    "img_features": img_features,
                                    "text_features": text_features
                                    }

        return dct_with_representations


    def compose_img_text(self, img, text_query):
        """
        """
        img_features = self.extract_img_feature(img)
        text_features = self.extract_text_feature(text_query)

        return self.compose_img_text_features(img_features, text_features)


    def compute_soft_triplet_loss_(self, mod_img1, img2):
        """
        """
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for anchor in range(len(labels)):
            triplets_anchor = []
            for positive in range(len(labels)):
                if labels[anchor] == labels[positive] and anchor != positive:
                    for negative in range(len(labels)):
                        if labels[anchor] != labels[negative]:
                            triplets_anchor.append([anchor, positive, negative])
            np.random.shuffle(triplets_anchor)
            triplets += triplets_anchor[:3]

        assert (triplets and len(triplets) < 2000)

        triplets = torch.as_tensor(np.asarray(triplets)).to(self.device)
        return self.triplet_loss(torch.cat([mod_img1, img2]), triplets)


    def compute_loss(self, imgs_query, text_query, imgs_target):
        """
        """

        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        composed_source_img = self.normalization_layer(dct_with_representations["repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm)

        assert (composed_source_img.shape[0] == target_img_features.shape[0] and
                composed_source_img.shape[1] == target_img_features.shape[1])

        conjugate_representations = self.compose_img_text_features(target_img_features_non_norm, dct_with_representations["text_features"],\
                                                        CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False))
        composed_target_img = self.normalization_layer(conjugate_representations["repres"])
        source_img_features = self.normalization_layer(dct_with_representations["img_features"])

        dct_with_representations["rot_sym_loss"] = \
                self.compute_soft_triplet_loss_(composed_target_img, source_img_features)

        return self.compute_soft_triplet_loss_(composed_source_img,\
                                                   target_img_features), dct_with_representations
