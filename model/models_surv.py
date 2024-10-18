from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

from util.options import get_args_parser_finetune
args = get_args_parser_finetune()
args = args.parse_args()
num_omics = args.num_omics

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # MAE encoder multi-omics multimodal
        self.encoder_omics = nn.Linear(300, self.embed_dim, bias=True)

        self.pos_embed = nn.Parameter(torch.zeros(1,
                                  (1024 // self.patch_embed.patch_size[0]) ** 2 + 1 + num_omics, self.embed_dim,),
                                    requires_grad=False)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            self.fc_norm1 = norm_layer(embed_dim)
            self.fc_norm2 = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.classifier1 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.classifier2 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.classifier3 = nn.Sequential(nn.Linear(768*2, 1), nn.Sigmoid())
        self.classifier4 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.classifier5 = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

    def forward_features(self, samples):
        x, X_mrna, X_mirna, X_meth = samples[:]

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1 + num_omics:, :]

        # append omics token
        omics = torch.cat((X_mrna.unsqueeze(1), X_mirna.unsqueeze(1), X_meth.unsqueeze(1)), dim=1)
        omics = self.encoder_omics(omics)
        X_omics = omics + self.pos_embed[:, 1:num_omics+1, :].expand(x.shape[0], -1, -1)
        x = torch.cat((X_omics, x), dim=1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x1 = x[:,1:num_omics+1,:].mean(dim=1)  # global pool with omics tokens
        embed_omics = self.fc_norm(x1)

        x2 = x[:, num_omics+1:, :].mean(dim=1)  # global pool with image tokens
        embed_image = self.fc_norm(x2)

        x3 = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        embed_all = self.fc_norm(x3)

        x4 = x[:, 0, :]  # cls token
        embed_cls = self.fc_norm(x4)

        return embed_omics, embed_image, embed_all, embed_cls

    def forward(self, x):
            embed_omics, embed_image, embed_all, embed_cls = self.forward_features(x)

            # survival risk score computing methods

            # method 1
            # x1 = self.classifier1(embed_omics)
            # x2 = self.classifier2(embed_image)
            # risk_score = (x1 + x2) / 2

            # method 2
            risk_score = self.classifier3(torch.cat((embed_omics, embed_image), dim=1))

            # method 3
            # risk_score = self.classifier4(embed_all)

            # method 4
            # risk_score = self.classifier5(embed_cls)

            # method 5
            # risk_score = torch.nn.functional.cosine_similarity(embed_omics, embed_image)
            # risk_score = risk_score.unsqueeze(1)

            return risk_score


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=1024, patch_size=128, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model