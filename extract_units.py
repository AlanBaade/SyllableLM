# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from syllablelm.data2vec.data.modality import Modality
from syllablelm.data2vec.models.data2vec2 import Data2VecMultiModel
from types import SimpleNamespace

THRESHOLD = 1 / .10 / 50.
FULL_MODELS_DICT = {
    '8.33Hz': {
        'delta': 0.0033,
        'quantile': 0.75,
    },
    '6.25Hz': {
        'delta': 0.0028,
        'quantile': 0.75,
    },
    '5.0Hz': {
        'delta': 0.0019,
        'quantile': 0.75,
    },
}

d2v2_config = SimpleNamespace(
    **{'_name': 'data2vec_multi', 'loss_beta': 0.0, 'loss_scale': None, 'depth': 8, 'start_drop_path_rate': 0.0, 'end_drop_path_rate': 0.0, 'num_heads': 12,
       'norm_eps': 1e-05, 'norm_affine': True, 'encoder_dropout': 0.1, 'post_mlp_drop': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0,
       'dropout_input': 0.0, 'layerdrop': 0.05, 'embed_dim': 768, 'mlp_ratio': 4.0, 'layer_norm_first': False, 'average_top_k_layers': 8,
       'end_of_block_targets': False, 'clone_batch': 8, 'layer_norm_target_layer': False, 'batch_norm_target_layer': False, 'instance_norm_target_layer': True,
       'instance_norm_targets': False, 'layer_norm_targets': False, 'ema_decay': 0.999, 'ema_same_dtype': True, 'log_norms': True, 'ema_end_decay': 0.99999,
       'ema_anneal_end_step': 75000, 'ema_encoder_only': False, 'max_update': 400000, 'modalities': SimpleNamespace(**{'_name': None, 'audio': SimpleNamespace(
            **{'type': Modality.AUDIO, 'prenet_depth': 4, 'prenet_layerdrop': 0.05, 'prenet_dropout': 0.1, 'start_drop_path_rate': 0.0,
               'end_drop_path_rate': 0.0, 'num_extra_tokens': 0, 'init_extra_token_zero': True, 'mask_noise_std': 0.01, 'mask_prob_min': None, 'mask_prob': 0.5,
               'inverse_mask': False, 'mask_prob_adjust': 0.05, 'keep_masked_pct': 0.0, 'mask_length': 5, 'add_masks': False, 'remove_masks': False,
               'mask_dropout': 0.0, 'encoder_zero_mask': True, 'mask_channel_prob': 0.0, 'mask_channel_length': 64, 'ema_local_encoder': False,
               'local_grad_mult': 1.0, 'use_alibi_encoder': True, 'alibi_scale': 1.0, 'learned_alibi': False, 'alibi_max_pos': None,
               'learned_alibi_scale': True, 'learned_alibi_scale_per_head': True, 'learned_alibi_scale_per_layer': False, 'num_alibi_heads': 12,
               'model_depth': 8, 'decoder': SimpleNamespace(
                    **{'decoder_dim': 384, 'decoder_groups': 16, 'decoder_kernel': 7, 'decoder_layers': 4, 'input_dropout': 0.1, 'add_positions_masked': False,
                       'add_positions_all': False, 'decoder_residual': True, 'projection_layers': 1, 'projection_ratio': 2.0,
                       'channel_mult': [1, 0.5, 0.25, 0.25, 0.25], 'decoder_transformer_layers': 4}), 'extractor_mode': 'layer_norm',
               'feature_encoder_spec': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_pos_width': 95, 'conv_pos_groups': 16,
               'conv_pos_depth': 5, 'conv_pos_pre_ln': False})}), 'shared_decoder': None, 'min_target_var': 0.1, 'min_pred_var': 0.01,
       'supported_modality': Modality.AUDIO, 'mae_init': False, 'seed': 1, 'skip_ema': False, 'cls_loss': 0.0, 'recon_loss': 0.0, 'd2v_loss': 1.0,
       'decoder_group': False}
)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.cluster_centers = np.load(km_path)
        self.C_np = self.cluster_centers.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        self.C = self.C.cuda()
        self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        dist = (
                x.pow(2).sum(-1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
        )
        return dist.argmin(dim=-1).cpu()


# @torch.compile()
@torch.inference_mode()
def efficient_extraction_dp_helper(x, threshold=THRESHOLD, s=35, min_hop=3):
    # x: batch, num feats, dimension feature vector. No support for padding, but setting remaining feats to zeros probably works
    # threshold: max hz before search using delta
    # s: max size of a chunk (50=1sec). 50 for librispeech, 35 for librilight
    # min_hop: min size of a chunk
    # Alan: If you have questions, I'm sorry. I forget how this works, too.

    b, n, d = x.shape

    dists = x.new_full((b, s + 1, n + s), 16384)

    rolled = torch.stack([torch.roll(x, shifts=-i, dims=-2) for i in range(s)]).transpose(0, 1)
    rolled_prepend = x[:, :s].unsqueeze(2).repeat(1, 1, s - 1, 1)
    arranged = torch.cat([rolled_prepend, rolled], dim=2)

    len_indices = torch.arange(s, device=x.device) + 1
    dots = arranged.pow(2).mean(dim=-1).cumsum(dim=-2)
    middle = -1 / len_indices.view(1, -1, 1) * arranged.cumsum(dim=-3).pow(2).mean(dim=-1)
    outs = dots + middle
    outs = torch.cat([outs[:, i:i + 1].roll(shifts=-(s - i - 1), dims=2) for i in range(s)], dim=1)
    dists[:, 1:, s:] = outs[:, :, :-(s - 1)]
    dists += dists.new_full(dists.shape, 16384).tril(s - 2)
    dists = dists.clamp(max=16384)

    m = int(threshold * n)
    total_dists = x.new_full((b, n + 2), 16384)
    total_dists[:, 0] = 0
    back = x.new_zeros((b, n + 1, m + 1), dtype=int)
    magic_mask = torch.tensor(
        [[(j + 1 - k if j + 1 >= k else n + 1) for j in range(n)] for k in range(min_hop, s + 1)], device=x.device
    ).unsqueeze(0).expand(b, s + 1 - min_hop, n)

    for j in range(1, m + 1):
        cur_min = torch.min(total_dists.unsqueeze(1).expand(b, s + 1 - min_hop, n + 2).gather(2, magic_mask) + dists[:, min_hop:, s:n + s], dim=1)
        total_dists[:, 1:-1] = cur_min.values
        back[:, 1:1 + n, j] = cur_min.indices + min_hop

    return dists, back


def get_quantile_borders_helper(dists, back, n=None, s=None, num_units=None, delta=None, quantile=None):
    # Binary search on dp array for the dynamic number of cuts given delta. Section 5.3 of paper.

    min_, max_ = num_units // 3, num_units
    best_m = min_

    while min_ <= max_:
        mid_ = (min_ + max_) // 2

        q = n
        j = mid_
        costs = []
        while q > 0:
            costs.append(dists[back[q, j], q - 1 + s] / back[q, j])
            q = (q - back[q, j])
            j = j - 1
        quantile_cost = np.quantile(costs, quantile)

        if quantile_cost > delta:
            min_ = mid_ + 1
            best_m = mid_
        else:
            max_ = mid_ - 1

    q = n
    j = best_m
    borders = [q]
    while q > 0:
        q = (q - back[q, j])
        borders.append(q)
        j = j - 1
    borders.reverse()

    return borders


@torch.no_grad()
def efficient_extraction(embeddings, threshold=THRESHOLD, s=35, min_hop=3, deltas=None, quantiles=None):
    b, n, d = embeddings.shape
    x = embeddings.cuda().float()
    m = int(threshold * n)
    s = min(n, s)

    dists, back = efficient_extraction_dp_helper(
        x, threshold=threshold, s=s, min_hop=min_hop
    )

    back = back.cpu().numpy()
    dists = dists.cpu().numpy()

    batch_outs = [[get_quantile_borders_helper(d_, b_, n=n, s=s, num_units=m, delta=delta, quantile=quantile)
                   for d_, b_ in zip(dists, back)] for delta, quantile in zip(deltas, quantiles)]

    return batch_outs


class SylBoostFeatureReader:
    def __init__(
            self,
            sylboost_checkpoint,
            kmeans_centroids_path,
            agglom_indices_path,
            model_key,
    ):
        d2v2_model = Data2VecMultiModel(d2v2_config, [Modality.AUDIO])
        d2v2_model = d2v2_model.cuda().eval().half()
        state_dict = torch.load(sylboost_checkpoint)
        d2v2_model.load_state_dict({k[len('model.'):]: v for k, v in state_dict['model_seg'].items()})
        self.d2v2_model = d2v2_model

        self.kmeans_centroids = ApplyKmeans(kmeans_centroids_path)
        self.agglom = np.load(agglom_indices_path)
        self.model_key = model_key

        assert model_key in FULL_MODELS_DICT.keys()
        self.delta = FULL_MODELS_DICT[model_key]['delta']
        self.quantile = FULL_MODELS_DICT[model_key]['quantile']

    @torch.no_grad()
    def forward(
            self,
            x,
    ):
        # Input:
        # x : (b, t) batched waveform tensor at 16000Hz.
        # Returns:
        # features: (b, n, d) raw data2vec2 features
        # clusters_with_times: list of length b, each item is clusters and boundaries. Clusters has 3 rows.
        #   # 0th row: KMeans+Agglom Cluster. 1st row: start boundary idx (inclusive). 2nd row: end boundary idx (exclusive).

        features = self.d2v2_model(x.half(), mode=None, mask=False, features_only=True, remove_extra_tokens=True, out_layer=-2)['x']
        result = {
            'features': features,
            'clusters_with_times': [],
        }

        # Multiple deltas at once suported (why not) but we just use one
        deltas = [self.delta]
        quantiles = [self.quantile]
        mincut = efficient_extraction(features, deltas=deltas, quantiles=quantiles)[0]

        for b_idx, (feats, mincut_boundaries) in enumerate(zip(features, mincut)):
            mincut_boundaries = np.array(mincut_boundaries)
            meaned_features = torch.stack([
                feats[mincut_boundaries[idx] + 1:mincut_boundaries[idx + 1] - 1].mean(dim=0)
                for idx in range(len(mincut_boundaries) - 1)
            ])  # t,dim
            meaned_features = (meaned_features - meaned_features.mean(dim=-1, keepdim=True)) / meaned_features.std(dim=-1, keepdim=True)

            clusters = self.agglom[self.kmeans_centroids(meaned_features.float()).numpy()].reshape(-1)

            # Sequential Deduplication
            not_repeat_mask = ~np.insert((clusters[1:] == clusters[:-1]), 0, 0)
            not_repeat_mask_end = ~np.insert((clusters[1:] == clusters[:-1]), clusters.shape[0] - 1, 0)  # RLE

            clusters_with_times = np.stack(
                [clusters[not_repeat_mask], mincut_boundaries[:-1][not_repeat_mask], mincut_boundaries[1:][not_repeat_mask_end]]
            )

            result['clusters_with_times'].append(clusters_with_times)

        return result


if __name__ == '__main__':
    sylboost_reader = SylBoostFeatureReader(
'/path/to/sylboost.pt'
        '/path/to/kmeans.npy',
        '/path/to/agglom.npy',
        '8.33Hz',
    )
    print(sylboost_reader.forward(torch.zeros(1, 48000).cuda().half()))
    """
    {'features': tensor([[[-0.2761, -0.0133, -0.1041,  ...,  0.2306,  0.0610, -0.1206],
         [-0.2634, -0.0099, -0.1146,  ...,  0.2805,  0.0734, -0.1403],
         [-0.1667, -0.0224, -0.1300,  ...,  0.3054,  0.0764, -0.1445],
         ...,
         [-0.1489, -0.1339, -0.2045,  ...,  0.0841, -0.0118, -0.2632],
         [-0.1512, -0.1378, -0.2325,  ...,  0.0895, -0.0026, -0.2671],
         [-0.1489, -0.0981, -0.2749,  ...,  0.0620,  0.0337, -0.2437]]],
       device='cuda:0', dtype=torch.float16), 'clusters_with_times': [array([[ 452,  104, 1107, 1008,  881,  415, 1132],
       [   0,   11,   20,   29,  106,  125,  142],
       [  11,   20,   29,  106,  125,  142,  149]])]}
    """
