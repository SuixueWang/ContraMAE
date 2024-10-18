from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import torch.nn.functional as F

from util.options import get_args_parser_pretrain
args = get_args_parser_pretrain()
args = args.parse_args()
num_omics = args.num_omics

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=1024, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # HIMOP encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_omics, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # HIMOP encoder multi-omics multimodal
        self.encoder_omics = nn.Linear(300, embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # HIMOP decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_omics, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_image = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to image
        self.decoder_omics = nn.Linear(decoder_embed_dim, 300, bias=True)  # decoder to omics
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True, omics_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True, omics_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, img, omics, mask_ratio):
        # embed patches
        x = self.patch_embed(img)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1+num_omics:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        omi, mask_o, ids_restore_o = self.random_masking(omics, mask_ratio=0.33)

        # append omics token
        X_omics = self.encoder_omics(omi)
        X_omics = X_omics + self.pos_embed[:, 1:num_omics, :].expand(x.shape[0], -1, -1)
        x = torch.cat((X_omics, x), dim=1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, mask_o, ids_restore_o

    def forward_decoder(self, x, ids_restore, mask_ratio, mask_o, ids_restore_o):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to image
        mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore.shape[1] * mask_ratio), 1)
        img = torch.cat([x[:, 1+num_omics-1:, :], mask_tokens], dim=1)  # no cls token
        img = torch.gather(img, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # append mask tokens to omics
        mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore_o.shape[1] * 2 // 3), 1)
        omi = torch.cat([x[:, 1:num_omics, :], mask_tokens], dim=1)  # no cls token
        omi = torch.gather(omi, dim=1, index=ids_restore_o.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, 0:1, :], omi, img], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x_omics = x[:, 1:1 + num_omics, :]
        x_image = x[:, 1 + num_omics:, :]

        # predictor projection
        pred_omics = self.decoder_omics(x_omics)
        pred_image = self.decoder_image(x_image)

        return pred_omics, pred_image

    def forward_loss_image(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward_loss_omics(self, pred_omics, omics, mask_o):
        # calculate the loss of masked omics modeling

        loss_mom = (pred_omics - omics) ** 2
        loss_mom = loss_mom.mean(dim=-1)  # [N, L], mean loss per patch

        loss_mom = (loss_mom * mask_o).sum() / mask_o.sum()  # mean loss on removed patches

        return loss_mom


    def forward_loss_contrastive(self, latent):
        """
        # calculating contrastive loss by referring CLIP code
        """

        omics_feats = latent[:, 1:4, :].mean(dim=1, keepdim=False)
        image_feats = latent[:, 5:, :].mean(dim=1, keepdim=False)

        # normalized features
        omics_features = omics_feats / omics_feats.norm(dim=1, keepdim=True)
        image_features = image_feats / image_feats.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        sim_i2o = logit_scale * image_features @ omics_features.t()
        sim_o2i = sim_i2o.t()
        contrastive_labels = torch.arange(latent.shape[0], device=latent.device)
        loss_contr = (F.cross_entropy(sim_i2o, contrastive_labels) + \
                    F.cross_entropy(sim_o2i, contrastive_labels)) / 2

        return loss_contr


    def forward(self, img, X_mrna, X_mirna, X_meth, mask_ratio=0.75):
        omics = torch.cat((X_mrna.unsqueeze(1), X_mirna.unsqueeze(1), X_meth.unsqueeze(1)), dim=1)
        latent, mask, ids_restore, mask_o, ids_restore_o = self.forward_encoder(img, omics, mask_ratio)
        pred_omics, pred_image = self.forward_decoder(latent, ids_restore, mask_ratio, mask_o, ids_restore_o)  # [N, L, p*p*3]
        loss_image = self.forward_loss_image(img, pred_image, mask)
        loss_omics = self.forward_loss_omics(pred_omics, omics, mask_o)
        loss_contr = self.forward_loss_contrastive(latent)

        loss = loss_omics * 1 + loss_image * 1 + loss_contr * 1

        return loss, loss_omics, loss_image, loss_contr


def himop_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=128, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def himop_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def himop_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
himop_vit_base_patch16 = himop_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
himop_vit_large_patch16 = himop_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
himop_vit_huge_patch14 = himop_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
