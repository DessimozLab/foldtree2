import torch
import pytorch_lightning as L


class QuaternionFoldingRefiner(torch.nn.Module):
    """
    Standalone iterative folding refinement module.

    This module is designed to be trained separately after a smaller encoder/
    decoder has been trained. It consumes residue embeddings plus an initial
    quaternion+translation state, and refines geometry iteratively.

    Initial state sources (priority order):
    1. `init_rt` / `init_quat` + `init_trans`
    2. `decoder_out` dict with `rt_pred` or (`quat_pred`, `trans_pred`)
    3. identity quaternions + zero translations

    Stopping criteria:
    - maximum refinement steps
    - convergence tolerance on mean state change after minimum steps
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout_p=0.05,
        decoder_hidden=128,
        out_angles=False,
        nheads=8,
        nlayers=3,
        max_refinement_steps=4,
        convergence_tol=1e-4,
        min_refinement_steps=1,
    ):
        super().__init__()

        self.args = locals()
        self.args.pop("self")

        L.seed_everything(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.out_angles = out_angles
        self.max_refinement_steps = max_refinement_steps
        self.convergence_tol = convergence_tol
        self.min_refinement_steps = min_refinement_steps

        d_model = hidden_channels[0] if isinstance(hidden_channels, (list, tuple)) else hidden_channels
        ff_dim = d_model * 2

        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.state_dim = 7
        self.input2transformer = torch.nn.Sequential(
            torch.nn.Linear(in_channels + self.state_dim, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
        )

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.delta_quat_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, decoder_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(decoder_hidden, 4),
        )
        self.delta_trans_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, decoder_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(decoder_hidden, 3),
        )

        self.angle_out = None
        if out_angles:
            self.angle_out = torch.nn.Sequential(
                torch.nn.Linear(d_model, decoder_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(decoder_hidden, out_channels),
                torch.nn.Tanh(),
            )

    @staticmethod
    def _normalize_quat(q):
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _pack_by_batch(x, batch):
        if batch is None:
            mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=x.device)
            return x.unsqueeze(0), mask

        num_graphs = int(batch.max().item()) + 1
        x_split = [x[batch == i] for i in range(num_graphs)]
        max_len = max(xi.shape[0] for xi in x_split)

        padded, mask = [], []
        for xi in x_split:
            orig_len = xi.shape[0]
            pad_len = max_len - orig_len
            if pad_len > 0:
                pad = torch.zeros(pad_len, xi.shape[1], device=xi.device, dtype=xi.dtype)
                xi = torch.cat([xi, pad], dim=0)
            padded.append(xi)
            m = torch.zeros(max_len, dtype=torch.bool, device=xi.device)
            m[:orig_len] = True
            mask.append(m)

        return torch.stack(padded, dim=0), torch.stack(mask, dim=0)

    @staticmethod
    def _unpack_by_batch(x_padded, batch, mask):
        if batch is None:
            return x_padded.squeeze(0)

        out = []
        for i in range(x_padded.shape[0]):
            out.append(x_padded[i][mask[i]])
        return torch.cat(out, dim=0)

    def _get_initial_rt(
        self,
        n_nodes,
        device,
        dtype,
        decoder_out=None,
        init_rt=None,
        init_quat=None,
        init_trans=None,
    ):
        quat = None
        trans = None

        if decoder_out is not None and isinstance(decoder_out, dict):
            if decoder_out.get("rt_pred", None) is not None:
                rt = decoder_out["rt_pred"]
                quat = rt[..., :4]
                trans = rt[..., 4:]
            elif decoder_out.get("quat_pred", None) is not None and decoder_out.get("trans_pred", None) is not None:
                quat = decoder_out["quat_pred"]
                trans = decoder_out["trans_pred"]

        if init_rt is not None:
            quat = init_rt[..., :4]
            trans = init_rt[..., 4:]
        if init_quat is not None:
            quat = init_quat
        if init_trans is not None:
            trans = init_trans

        if quat is None:
            quat = torch.zeros(n_nodes, 4, device=device, dtype=dtype)
            quat[:, 0] = 1.0
        if trans is None:
            trans = torch.zeros(n_nodes, 3, device=device, dtype=dtype)

        quat = self._normalize_quat(quat.to(device=device, dtype=dtype))
        trans = trans.to(device=device, dtype=dtype)
        return quat, trans

    def forward(
        self,
        data,
        decoder_out=None,
        init_rt=None,
        use_self_output=True,
        return_history=False,
        **kwargs,
    ):
        if isinstance(data, dict):
            x_dict = data
            batch = kwargs.get("batch", None)
        else:
            x_dict = data.x_dict
            batch = data["res"].batch if hasattr(data["res"], "batch") else None

        x_res = self.bn(x_dict["res"])
        x_res = self.dropout(x_res)

        n_nodes = x_res.shape[0]
        quat, trans = self._get_initial_rt(
            n_nodes=n_nodes,
            device=x_res.device,
            dtype=x_res.dtype,
            decoder_out=decoder_out,
            init_rt=init_rt,
            init_quat=kwargs.get("init_quat", None),
            init_trans=kwargs.get("init_trans", None),
        )

        max_steps = max(1, int(kwargs.get("max_refinement_steps", self.max_refinement_steps)))
        tol = float(kwargs.get("convergence_tol", self.convergence_tol))
        min_steps = int(kwargs.get("min_refinement_steps", self.min_refinement_steps))

        history = []
        last_change = None
        steps_done = 0
        converged = False
        encoded_flat = None

        for step in range(max_steps):
            state = torch.cat([quat, trans], dim=-1)
            xt = torch.cat([x_res, state], dim=-1)
            xt = self.input2transformer(xt)

            x_padded, valid_mask = self._pack_by_batch(xt, batch)
            encoded = self.transformer_encoder(x_padded, src_key_padding_mask=~valid_mask)
            encoded_flat = self._unpack_by_batch(encoded, batch, valid_mask)

            delta_q = self.delta_quat_head(encoded_flat)
            delta_t = self.delta_trans_head(encoded_flat)

            quat_new = self._normalize_quat(quat + delta_q)
            trans_new = trans + delta_t

            change_q = torch.norm(quat_new - quat, dim=-1).mean()
            change_t = torch.norm(trans_new - trans, dim=-1).mean()
            change = (change_q + change_t).item()

            if return_history:
                history.append(
                    {
                        "step": step + 1,
                        "mean_quat_change": float(change_q.item()),
                        "mean_trans_change": float(change_t.item()),
                        "mean_total_change": float(change),
                    }
                )

            quat = quat_new
            trans = trans_new
            last_change = change
            steps_done = step + 1

            if step + 1 >= min_steps and change < tol:
                converged = True
                break

            if not use_self_output and step == 0:
                break

        rt_pred = torch.cat([quat, trans], dim=-1)
        angles = None
        if self.angle_out is not None and encoded_flat is not None:
            angles = self.angle_out(encoded_flat) * torch.pi

        result = {
            "quat_pred": quat,
            "trans_pred": trans,
            "rt_pred": rt_pred,
            "angles": angles,
            "refinement_steps": steps_done,
            "refinement_converged": converged,
            "refinement_mean_change": float(last_change) if last_change is not None else 0.0,
        }

        if return_history:
            result["refinement_history"] = history

        return result
