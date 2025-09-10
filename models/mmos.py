import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffattention.multihead_flashdiff import MultiheadFlashDiff
from models.utils import PreNorm, Attention, FeedForward


class MMOS(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_size=768,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
        num_kv_heads=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_size), requires_grad=True
        )
        self.text_patch_token = nn.Parameter(
            torch.randn(1, 1, hidden_size), requires_grad=True
        )
        self.text_slide_token = nn.Parameter(
            torch.randn(1, 1, hidden_size), requires_grad=True
        )
        self.image_patch_token = nn.Parameter(
            torch.randn(1, 1, hidden_size), requires_grad=True
        )

        self.text_slide_layers = nn.ModuleList(
            [
                nn.Sequential(
                    PreNorm(
                        hidden_size,
                        Attention(hidden_size, heads=1, dropout=dropout),
                        context_dim=hidden_size,
                    ),
                    FeedForward(hidden_size, mult=1, dropout=dropout),
                )
                for i in range(num_layers)
            ]
        )
        self.text_patch_layers = nn.ModuleList(
            [
                nn.Sequential(
                    PreNorm(
                        hidden_size,
                        Attention(hidden_size, heads=1, dropout=dropout),
                        context_dim=hidden_size,
                    ),
                    FeedForward(hidden_size, mult=1, dropout=dropout),
                )
                for i in range(num_layers)
            ]
        )
        self.image_patch_layer = nn.ModuleList(
            [
                self._build_transformer_layer(
                    hidden_size, i + 1, num_heads, num_kv_heads, dropout
                )
                for i in range(num_layers)
            ]
        )
        self.image_slide_layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(num_layers)]
        )

        self.image_patch_aggregator = nn.Sequential(
            PreNorm(
                hidden_size,
                Attention(hidden_size, heads=1, dropout=dropout),
                context_dim=hidden_size,
            ),
            FeedForward(hidden_size, mult=1, dropout=dropout),
        )

        self.cls_head = PreNorm(
                hidden_size,
                Attention(hidden_size, heads=1, dropout=dropout),
                context_dim=hidden_size)

        self.to_output = nn.Linear(hidden_size, num_classes)

    def _build_transformer_layer(
        self,
        hidden_size: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
    ) -> nn.Module:
        """Construct a transformer layer with residual connection"""
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            MultiheadFlashDiff(hidden_size, layer_idx, num_heads, num_kv_heads),
            nn.Dropout(dropout),
        )

    def forward(self, text_base, text_patch, text_slide, image_patch, image_slide):
        """
        Args:
            text_base: (batch_size, seq_len, hidden_size)
            text_patch: (batch_size, seq_len, hidden_size)
            text_slide: (batch_size, seq_len, hidden_size)
            image_patch: (batch_size, num_patch, hidden_size)
            image_slide: (batch_size, 1, hidden_size)
        Returns:
            logits: (batch_size, num_classes)
        """

        text_patch = torch.concat([self.text_patch_token, text_base, text_patch], dim=1)
        text_slide = torch.concat([self.text_slide_token, text_slide], dim=1)

        for layer_i in range(self.num_layers):
            text_patch_residual = text_patch
            text_slide_residual = text_slide
            image_patch_residual = image_patch
            image_slide_residual = image_slide
            text_patch, _ = self.text_patch_layers[layer_i][0](
                text_patch, context=image_patch
            )
            text_patch = self.text_patch_layers[layer_i][1](text_patch)
            text_slide, _ = self.text_slide_layers[layer_i][0](
                text_slide, context=image_slide
            )
            text_slide = self.text_slide_layers[layer_i][1](text_slide)
            image_patch = self.image_patch_layer[layer_i](image_patch)
            image_slide = self.image_slide_layer[layer_i](image_slide)
            text_patch = text_patch + text_patch_residual
            text_slide = text_slide + text_slide_residual
            image_patch = image_patch + image_patch_residual
            image_slide = image_slide + image_slide_residual

        image_patch_token, _ = self.image_patch_aggregator[0](
            self.image_patch_token, context=image_patch
        )
        image_patch_token = self.image_patch_aggregator[1](image_patch_token)
        image_slide_token = image_slide
        text_patch_token = text_patch[:, 0, :].unsqueeze(1)
        text_slide_token = text_slide[:, 0, :].unsqueeze(1)
        unified_tokens = torch.concat(
            [text_patch_token, text_slide_token, image_patch_token, image_slide_token],
            # [text_patch_token, text_slide_token],
            dim=1,
        )
        cls_token, _ = self.cls_head(self.cls_token, context=unified_tokens)
        logits = self.to_output(cls_token)

        return logits


# if __name__ == "__main__":
#     with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
#         model = MMOS(10).cuda()
#         text = torch.randn(1, 55, 768).cuda()
#         slide_image = torch.randn(1, 1, 768).cuda()
#         patch_image = torch.randn(1, 1000, 768).cuda()
#         logits = model(text, slide_image, patch_image)
#     print(logits.shape)
