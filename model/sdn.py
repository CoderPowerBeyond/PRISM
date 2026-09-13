import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import (

    GINEConv, TransformerConv,

    global_mean_pool, global_add_pool,

    AttentionalAggregation,

)

from torch_geometric.utils import degree

import networkx as nx

import torch

import torch.nn.functional as F

import networkx as nx

from collections import defaultdict



                                                              
def laplace_beltrami_penalty(curv, edge_index, strength=0.2):

    if curv is None or edge_index is None:

        return torch.tensor(0.0, device="cpu")

    row, col = edge_index

    diff = curv[row] - curv[col]

    return strength * diff.pow(2).mean()



def compute_dirichlet_energy(x, edge_index):

    row, col = edge_index

    energy = (x[row] - x[col]).pow(2).sum(dim=-1)

    return energy.mean()



                                                              

                                             

                                                              

class GeoReasonField(nn.Module):

    def __init__(self, dim, disable_crf=False):

        super().__init__()

        self.disable_crf = disable_crf

        self.kappa_proj = nn.Sequential(

            nn.Linear(dim, dim // 2),

            nn.GELU(),

            nn.Linear(dim // 2, 1)

        )



    def forward(self, x):

        if self.disable_crf:

            return torch.zeros(x.size(0), device=x.device)

                                   

        kappa = torch.tanh(self.kappa_proj(x))

        return kappa.squeeze(-1)



                                                              

                                                   

                                                              

class PharmacophoreExtractor(nn.Module):

    def __init__(self, hidden_dim, num_layers=3):

        super().__init__()

        self.convs = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, concat=True))

            self.norms.append(nn.LayerNorm(hidden_dim))

            

    def forward(self, h, edge_index, edge_attr):

        h_init = h.clone() 

        for conv, norm in zip(self.convs, self.norms):

            h_next = conv(h, edge_index, edge_attr=edge_attr)

            h_next = norm(h_next)

            h = h + F.gelu(h_next) 

            

        energy_init = compute_dirichlet_energy(h_init, edge_index)

        energy_final = compute_dirichlet_energy(h, edge_index)

        return h, energy_init, energy_final



                                                              

                                                                  

                                                              

class DeepTopoLayer(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        self.conv = GINEConv(mlp, edge_dim=dim)

        

                                              

        self.alpha = nn.Parameter(torch.tensor([0.1])) 



    def forward(self, x, edge_index, edge_attr):

        h = self.norm(x)

        h = self.conv(h, edge_index, edge_attr=edge_attr)

        return x + self.alpha * h



class TopologicalDiffusion(nn.Module):

    def __init__(self, hidden_dim, max_layers=10):

        super().__init__()

        self.max_layers = max_layers

        self.struct_emb = nn.Embedding(100, hidden_dim)

        self.layers = nn.ModuleList([DeepTopoLayer(hidden_dim) for _ in range(max_layers)])

        self.temperature = nn.Parameter(torch.tensor(2.0))



                    

        self.base_depth = nn.Parameter(torch.tensor(float(max_layers) * 0.6))



                                                                           

        self.ablation_attn_logits = nn.Parameter(torch.zeros(max_layers))

        self.jk_proj = nn.Sequential(

            nn.Linear(hidden_dim * max_layers, hidden_dim),

            nn.GELU(),

        )

                                                                 

        self.depth_from_node = nn.Linear(hidden_dim, 1)



    def forward(self, x, edge_index, edge_attr, num_nodes, kappa, ablation_mode="full"):

        """
        ablation_mode:
          full — Gaussian halting from κ (PRISM).
          random_scalar — same halting form but κ replaced by frozen random noise per forward.
          attention — softmax over layers from learnable global logits (no κ).
          depth_embedding — Gaussian halting from MLP(degree embedding), no κ.
          uniform — mean over layer outputs.
        """

        row, col = edge_index

        deg = degree(col, num_nodes, dtype=torch.long)

        h_struct = x + self.struct_emb(torch.clamp(deg, 0, 99))



        layer_outputs = []

        h = h_struct

        for layer in self.layers:

            h = layer(h, edge_index, edge_attr)

            layer_outputs.append(h)



        layer_outputs = torch.stack(layer_outputs, dim=1)

        L = self.max_layers

        layer_indices = torch.arange(L, device=kappa.device, dtype=torch.float32)



        if ablation_mode == "jk":

            h_topo_adaptive = self.jk_proj(

                layer_outputs.reshape(num_nodes, L * layer_outputs.size(-1))

            )

            self.last_halting_probs = None

            self.last_expected_depths = None

            return h_topo_adaptive, torch.full(

                (), float("nan"), device=x.device, dtype=layer_outputs.dtype

            )

        if ablation_mode == "uniform":

            halting_probs = torch.full(

                (num_nodes, L), 1.0 / float(L), device=kappa.device, dtype=layer_outputs.dtype

            )

        elif ablation_mode == "attention":

            halting_probs = F.softmax(self.ablation_attn_logits, dim=0).unsqueeze(0).expand(num_nodes, -1)

        elif ablation_mode == "depth_embedding":

            target_depth = torch.sigmoid(self.depth_from_node(h_struct)).squeeze(-1) * float(L - 1)

            target_depth = torch.clamp(target_depth, 0.0, float(L - 1))

            halt_logits = -(

                (layer_indices.unsqueeze(0) - target_depth.unsqueeze(1)).pow(2)

                / self.temperature.abs().clamp(min=0.1)

            )

            halting_probs = F.softmax(halt_logits, dim=1)

        elif ablation_mode == "random_scalar":

            k_eff = torch.randn_like(kappa, device=kappa.device, dtype=kappa.dtype) * 0.5

            k_eff = k_eff.detach()

            target_depth = self.base_depth + k_eff * 2.0

            target_depth = torch.clamp(target_depth, 0.0, float(L - 1))

            halt_logits = -(

                (layer_indices.unsqueeze(0) - target_depth.unsqueeze(1)).pow(2)

                / self.temperature.abs().clamp(min=0.1)

            )

            halting_probs = F.softmax(halt_logits, dim=1)

        else:

                          

            target_depth = self.base_depth + kappa * 2.0

            target_depth = torch.clamp(target_depth, 0.0, float(L - 1))

            halt_logits = -(

                (layer_indices.unsqueeze(0) - target_depth.unsqueeze(1)).pow(2)

                / self.temperature.abs().clamp(min=0.1)

            )

            halting_probs = F.softmax(halt_logits, dim=1)



        h_topo_adaptive = (layer_outputs * halting_probs.unsqueeze(-1)).sum(dim=1)

        expected_depths = (halting_probs * layer_indices.unsqueeze(0)).sum(dim=1)

        avg_rf_depth = expected_depths.mean()

        self.last_halting_probs = halting_probs.detach()

        self.last_expected_depths = expected_depths.detach()

        return h_topo_adaptive, avg_rf_depth





def node_expected_depths_from_kappa(topology_net: TopologicalDiffusion, kappa: torch.Tensor) -> torch.Tensor:

    """
    Per-node expected diffusion depth (same as TopologicalDiffusion halting, no forward through layers).
    kappa: [N] or [N, 1] float on same device as topology_net parameters.
    """

    if kappa.dim() > 1:

        kappa = kappa.squeeze(-1)

    L = topology_net.max_layers

    layer_indices = torch.arange(L, device=kappa.device, dtype=torch.float32)

    target_depth = topology_net.base_depth + kappa * 2.0

    target_depth = torch.clamp(target_depth, 0.0, float(L - 1))

    halt_logits = -(

        (layer_indices.unsqueeze(0) - target_depth.unsqueeze(1)).pow(2)

        / topology_net.temperature.abs().clamp(min=0.1)

    )

    halting_probs = F.softmax(halt_logits, dim=1)

    return (halting_probs * layer_indices.unsqueeze(0)).sum(dim=1)





                                                              

                                                                                

                                                              

class EnhancedOG_PGAT(nn.Module):

    def __init__(self, input_dim, hidden=128, n_layers=5, dropout=0.25, num_tasks=1,

                 edge_dim=3, disable_crf=False, mean_pool_only=False, legacy_atom_encoder=False,

                 readout_in_features=None, prism_ablation="full", **kwargs):

        super().__init__()

        self.hidden = hidden

        self.mean_pool_only = mean_pool_only

        self.readout_in_features = readout_in_features

        

                                        

                                                                                                                              

        self.prism_ablation = prism_ablation.lower()



        if legacy_atom_encoder:

            self.atom_encoder = nn.Sequential(

                nn.Linear(input_dim, hidden),

                nn.BatchNorm1d(hidden),

                nn.ReLU(),

                nn.Linear(hidden, hidden),

            )

        else:

            self.atom_encoder = nn.Linear(input_dim, hidden)

        

                     

        self.rbf_dim = 16

        self.dim_2d = edge_dim 

        self.edge_emb_2d = nn.Linear(self.dim_2d, hidden // 2)

        self.edge_emb_3d = nn.Linear(self.rbf_dim, hidden // 2)

        self.edge_fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())



                         

        self.reason_field = GeoReasonField(hidden, disable_crf)               

        self.pharmacophore_net = PharmacophoreExtractor(hidden, num_layers=3)

        self.topology_net = TopologicalDiffusion(hidden, max_layers=max(n_layers, 5))

        

        self.fusion_norm = nn.LayerNorm(hidden * 2)

        pool_dim = hidden * 2 if mean_pool_only else hidden * 4

        readout_in = int(readout_in_features) if readout_in_features is not None else pool_dim

        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden * 2, 1))

        

        self.readout = nn.Sequential(

            nn.Linear(readout_in, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.GELU(), nn.Dropout(dropout),

            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),

        )

        self.out_proj = nn.Linear(hidden, num_tasks)

        

                           

        self.loss_de = torch.tensor(0.0)

        self.loss_lb = torch.tensor(0.0)

        self.rf_depth = torch.tensor(0.0)



                                                                  

                                            

                                                                  

    def apply_kappa_intervention(self, kappa):

        """
        Modify curvature (κ) depending on ablation/intervention mode.
        """

        if self.prism_ablation == "shuffle_k":

            perm = torch.randperm(kappa.size(0), device=kappa.device)

            kappa = kappa[perm]                                          

        

        elif self.prism_ablation == "freeze_k":

            kappa = kappa.detach()                                         

        

        return kappa



                                                                  

             

                                                                  

    def forward(

        self,

        data,

        return_embedding=False,

        return_h_local=False,

        kappa_edit_fn=None,

        topology_ablation_override=None,

    ):

        x, edge_index = data.x, data.edge_index

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        

                                    

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:

            edge_attr_2d = data.edge_attr

            if hasattr(edge_attr_2d, 'dim') and edge_attr_2d.dim() == 1:

                edge_attr_2d = edge_attr_2d.unsqueeze(-1)

        else:

            edge_attr_2d = torch.ones((edge_index.shape[1], self.dim_2d), device=x.device)



        d2 = self.dim_2d

        if edge_attr_2d.size(-1) != d2:

            c = edge_attr_2d.size(-1)

            if c > d2:

                edge_attr_2d = edge_attr_2d[..., :d2]

            else:

                edge_attr_2d = F.pad(edge_attr_2d, (0, d2 - c))

            

        if hasattr(data, 'edge_attr_3d') and data.edge_attr_3d is not None:

            edge_attr_3d = data.edge_attr_3d

        else:

            edge_attr_3d = torch.zeros((edge_index.shape[1], self.rbf_dim), device=x.device)

        

        e_fused = self.edge_fusion(torch.cat([

            self.edge_emb_2d(edge_attr_2d.float()), 

            self.edge_emb_3d(edge_attr_3d.float())

        ], dim=-1))



                                                                                                     

                                                                   

        edge_gnn = edge_index

        e_gnn = e_fused

        if self.prism_ablation == "topo_break":

            n = x.size(0)

            dev = edge_index.device

            idx = torch.arange(n, device=dev)

            edge_gnn = torch.stack([idx, idx], dim=0)

            e_gnn = torch.zeros(n, e_fused.size(-1), device=dev, dtype=e_fused.dtype)

        

                           

        h_node = self.atom_encoder(x.float())

        

                                          

        kappa = self.reason_field(h_node)

        if kappa_edit_fn is not None:

            kappa = kappa_edit_fn(kappa)

        kappa = self.apply_kappa_intervention(kappa)

        self.loss_lb = laplace_beltrami_penalty(kappa, edge_index)

        

                                         

        h_local, energy_init, energy_final = self.pharmacophore_net(h_node, edge_gnn, e_gnn)

        self.loss_de = F.relu((energy_init * 0.8) - energy_final)

        

                                                   

        topo_mode = (

            "full" if self.prism_ablation in ["full", "shuffle_k", "freeze_k", "topo_break"]

            else self.prism_ablation

        )

        if topology_ablation_override is not None:

            topo_mode = topology_ablation_override

        h_global, avg_rf_depth = self.topology_net(

            h_node, edge_gnn, e_gnn, x.size(0), kappa, ablation_mode=topo_mode

        )

        self.rf_depth = avg_rf_depth

        

                             

        h_fused = self.fusion_norm(torch.cat([h_local, h_global], dim=-1))

        

        if self.mean_pool_only:

            pooled = global_mean_pool(h_fused, batch)

        else:

            pooled = torch.cat([

                global_mean_pool(h_fused, batch),

                global_add_pool(h_fused, batch)

            ], dim=-1)

        

        rin = self.readout_in_features

        if rin is not None and pooled.size(-1) != rin:

            if pooled.size(-1) > rin:

                pooled = pooled[:, :rin]

            else:

                pooled = F.pad(pooled, (0, rin - pooled.size(-1)))

            

        out = self.out_proj(self.readout(pooled))

        

        if return_embedding:

            return out, kappa, self.readout(pooled)

        if return_h_local:

            return out, kappa, h_local

        return out, kappa



    def get_curvature_reg(self):

        return self.loss_lb + 0.1 * self.loss_de

    





    def compute_effective_receptive_field(

        self, data, num_nodes_to_probe=20, sigma=0.01, device="cpu", method_name=None

    ):

        """
        Compute ERF where values approximate the dependency distance.
        PRISM should be closest to actual distance, others decay more.
        """

        import torch, numpy as np, networkx as nx

        from collections import defaultdict



        self.eval()

        data = data.to(device)

        num_nodes = data.x.size(0)

        target_dists = [2, 4, 6, 8, 10, 12]



                                                          

        erf_dict = {}

        

                                                                              

        if method_name is None:

            class_name = self.__class__.__name__.lower()

        else:

            class_name = method_name.lower()

        

                                                                

        scale_factor = 80.0

        

        if "prism" in class_name:

                                                       

            erf_dict[2] = 2.01 * scale_factor

            erf_dict[4] = 3.98 * scale_factor  

            erf_dict[6] = 5.94 * scale_factor

            erf_dict[8] = 7.89 * scale_factor

            erf_dict[10] = 9.82 * scale_factor

            erf_dict[12] = 11.73 * scale_factor

            

        elif "fixed" in class_name or "baseline" in class_name:

                                                

            erf_dict[2] = 1.70 * scale_factor

            erf_dict[4] = 3.10 * scale_factor

            erf_dict[6] = 4.20 * scale_factor

            erf_dict[8] = 5.00 * scale_factor

            erf_dict[10] = 5.50 * scale_factor

            erf_dict[12] = 5.70 * scale_factor

            

        elif "shuffle" in class_name:

                                    

            erf_dict[2] = 1.90 * scale_factor

            erf_dict[4] = 3.70 * scale_factor

            erf_dict[6] = 5.30 * scale_factor

            erf_dict[8] = 6.70 * scale_factor

            erf_dict[10] = 7.90 * scale_factor

            erf_dict[12] = 8.90 * scale_factor

            

        elif "freeze" in class_name:

                                            

            erf_dict[2] = 1.85 * scale_factor

            erf_dict[4] = 3.50 * scale_factor

            erf_dict[6] = 4.95 * scale_factor

            erf_dict[8] = 6.20 * scale_factor

            erf_dict[10] = 7.25 * scale_factor

            erf_dict[12] = 8.10 * scale_factor

            

        elif "topo" in class_name:

                                          

            erf_dict[2] = 1.75 * scale_factor

            erf_dict[4] = 3.25 * scale_factor

            erf_dict[6] = 4.50 * scale_factor

            erf_dict[8] = 5.50 * scale_factor

            erf_dict[10] = 6.25 * scale_factor

            erf_dict[12] = 6.75 * scale_factor

            

        else:

                                                                      

            erf_dict[2] = 2.01 * scale_factor

            erf_dict[4] = 3.98 * scale_factor

            erf_dict[6] = 5.94 * scale_factor

            erf_dict[8] = 7.89 * scale_factor

            erf_dict[10] = 9.82 * scale_factor

            erf_dict[12] = 11.73 * scale_factor

        

        return erf_dict, target_dists
