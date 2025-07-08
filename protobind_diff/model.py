from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl
import logging
import huggingface_hub

from .ligands.rdkit_utils import validate_smile, calc_chem_desc, tanimoto_smiles
from .ligands.smiles_tokenizer import ChemformerTokenizer
from .noise_schedule import _sample_t, q_xt, _sample_categorical, LogLinearNoise
from .decoder_rope import Decoder_RoPE

logger = logging.getLogger("lightning")


class ModelGenerator(pl.LightningModule):
    """
    ProtoBind-Diff model with SMILES and ESM-2 protein encodings.
    """
    @staticmethod
    def get_exp_dir(
            exp_dir: str | None,
            output_dir: str,
            exp_dir_prefix: str,
            split: str
    ) -> Path:
        """Determines the experiment directory path."""
        if exp_dir:
            return Path(exp_dir)
        return Path(output_dir) / split / exp_dir_prefix

    def __init__(self, *args, **kwargs):
        """Initializes the Lightning Module, saves hyperparameters, and configures the model."""
        super().__init__()

        is_load = kwargs['load']
        if not is_load:
            self.save_hyperparameters()

        self.data_dir = Path(kwargs["data_dir"])
        exp_dir = kwargs.get('exp_dir', None)
        self.exp_dir = self.get_exp_dir(
            exp_dir=exp_dir,
            output_dir=kwargs["output_dir"],
            exp_dir_prefix=kwargs["exp_dir_prefix"],
            split=kwargs["split"]
        )

        self.configure_model_params(**kwargs)

    def configure_model_params(self, **kwargs):
        """Parses keyword arguments to configure the model, tokenizer, and training parameters."""

        self.learning_rate = kwargs.pop('learning_rate')
        self.weight_decay = float(kwargs.pop('weight_decay'))

        # Decoder params for masked diffusion
        decoder_params = {
            'nhead': kwargs['num_heads_decoder'],
            'n_layers': kwargs['num_decoder_layers'],
            'hidden_size': kwargs['decoder_hidd_dim'],
            'expand_feedforward': kwargs['expand_feedforward'],
            'decoder_name': kwargs['decoder_name'],
        }
        # Tokenizer params
        tokenizer_path = kwargs.get('tokenizer_path')
        if tokenizer_path:
            self.tokenizer = ChemformerTokenizer(filename=tokenizer_path)
        else:
            self.tokenizer = ChemformerTokenizer(filename=self.data_dir / f"{kwargs['tokenizer_json_name']}.json")

        # Masking params
        self.noise = LogLinearNoise()
        self.mask_index = self.tokenizer.mask_token_id

        # Sampler params
        self.model_length = 170
        self.noise_removal = True
        self.nucleus_p = 0.9
        self.eta = 0.1
        self.sampling_steps = 100
        self.time_conditioning = False

        self.return_attention = False

        self.model = ProtobindMaskedDiffusion(
            embedding_dim=kwargs['seq_embedding_dim'],
            mask_index=self.mask_index,
            vocab_size=self.tokenizer.vocab_size,
            decoder_params=decoder_params,
            dropout=kwargs['dropout'],
        )
        self.optimizer = kwargs.get('optimizer', 'Adam')

    def generate_mols(self, sequence: Tuple[torch.Tensor, torch.Tensor],
                      return_attention=False) -> Tuple[np.array, torch.Tensor,np.array]:
        """Generates and validates SMILES strings for a given protein sequence.

        This method calls the internal sampler, decodes the generated tokens into
        SMILES strings, and filters out any invalid molecules.

        Args:
            sequence (Tuple[torch.Tensor, torch.Tensor]): The conditioned protein sequence
                embedding and its length.
            return_attention (bool): Whether to return attention maps from the sampler.

        Returns:
            Tuple[np.array, torch.Tensor, np.array]: A tuple containing the valid SMILES
            strings, corresponding attention maps, and the mask of valid indices.
        """
        samples, attention = self._sample(sequence, return_attention=return_attention)
        text_samples = self.tokenizer.decode(samples.long())
        text_samples = np.array([validate_smile(smile) for smile in text_samples])

        mask_invalid = (text_samples != None) & (text_samples != '.') & (text_samples != '')
        text_samples = text_samples[mask_invalid]
        if attention is not None:
            attention = attention[mask_invalid]

        return text_samples, attention, mask_invalid

    def predict_step(self, batch, batch_idx):
        sequence, smiles, seq_id, smi_id = batch
        gen_samples, attention, mask_invalid = self.generate_mols(
            sequence, return_attention=self.return_attention)
        seq_id = seq_id[mask_invalid]
        return gen_samples, attention, seq_id

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # dataloader_idx to predict on several validation sets
        return self.common_step(batch, "val", batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_step(batch, "test", batch_idx)

    def common_step(self, batch, description, batch_idx, dataloader_idx=None):
        """Performs a common training, validation, or test step.

            This method takes a batch, applies noise according to the diffusion
            timestep, runs the model forward, calculates the loss, and logs metrics.

            Args:
                batch (Tuple): The input batch from the dataloader.
                description (str): The step description (e.g., 'train', 'val').
                batch_idx (int): The index of the batch.

            Returns:
                torch.Tensor: The calculated loss for the batch.
         """
        sequence, smiles, seq_id, smi_id = batch

        # Get data and apply noise
        X, length = smiles
        bs = X.shape[0]
        X = X.squeeze(-1)
        padding_mask = (X != 0).float()  # 0 is pad token id
        t = _sample_t(X.shape[0], X.device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = q_xt(X, move_chance, self.mask_index)
        xt = xt.unsqueeze(dim=2)
        smiles_t = (xt, length, None)

        pred_x, _ = self.model(sequence, smiles_t, sigma, padding_mask)
        total_loss = self.loss_mdlm(X.long(), pred_x, sigma, dsigma, padding_mask=None)

        if batch_idx % 50 == 0:
            tokens = pred_x.argmax(dim=-1) * padding_mask
            true_smiles = self.tokenizer.decode(X.long())
            pred_smiles = [smile for smile in self.tokenizer.decode(tokens)]
            pred_smiles_valid = [validate_smile(smile) for smile in pred_smiles]
            
            try:
                tanimoto = np.asarray([tanimoto_smiles(mol_pred, mol_ref) for mol_pred, mol_ref
                                   in zip(pred_smiles_valid, true_smiles) if mol_pred is not None])
                tanimoto_mean = np.mean(tanimoto) if len(tanimoto) > 0 else 0
                num_mols_valid = len(tanimoto)
            except:
                num_mols_valid = 0
                tanimoto_mean = 0.0

            self.log(f"{description}_tanimoto", tanimoto_mean, prog_bar=True,
                     on_epoch=True, sync_dist=True)
            self.log(f"{description}_perc_of_valid", num_mols_valid / bs * 100, prog_bar=True,
                     on_epoch=True, sync_dist=True)

        self.log(f"{description}_loss", total_loss, prog_bar=True, on_epoch=True,
                 sync_dist=True, batch_size=bs)
        return total_loss

    def configure_optimizers(self):
        if self.weight_decay > 0.:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def loss_mdlm(self, x_0, model_output, sigma, dsigma, padding_mask=None):
        """Loss for SUBS parameterization, continuous time case"""
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x_0[:, :, None]).squeeze(-1)

        loss = - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

        if padding_mask is not None:
            return (loss * padding_mask).sum() / padding_mask.sum()
        return loss.mean()
    
    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, sequence, x, t, dt, p_x0=None, conf=None,
                             return_attention=False):
        attention = None
        if t.ndim > 1:
            t = t.squeeze(-1)
        sigma_t, _ = self.noise(t)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        padding_mask = (x != 0).float()
        
        if p_x0 is None:
            p_x0, attention = self.model(sequence, (x.unsqueeze(dim=2), None, None), sigma_t,
                                         padding_mask, return_attention=return_attention)
            p_x0 = p_x0.exp()
            if self.nucleus_p < 1:
                sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cumulative_probs <= self.nucleus_p
                top_p_mask[..., 0] = True
                nucleus_probs = sorted_probs * top_p_mask
                nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
                p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
        
        assert move_chance_t.ndim == p_x0.ndim
        
        # Use remdm-cap sampler
        alpha_t = (1 - move_chance_t)[0].item()
        alpha_s = (1 - move_chance_s)[0].item()
        if alpha_t > 0:
            sigma = min(self.eta, (1 - alpha_s) / alpha_t)
        else:
            sigma = self.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)

        if torch.allclose(xs, x) and not self.time_conditioning:
            p_x0_cache = p_x0
        else:
            p_x0_cache = None

        return p_x0_cache, xs, conf, attention

    @torch.no_grad()
    def _sample(self, sequence, eps=1e-3, return_attention=False):
        """Generate samples from the model"""
        num_steps = self.sampling_steps
        bs = sequence[0].shape[0]
        x = self._sample_prior(bs, self.model_length).to(self.device)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        min_t = timesteps[-1].item()
        confident_score = - torch.ones_like(x, device=self.device) * torch.inf
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(bs, 1, device=self.device)
            p_x0_cache, x_next, confident_score, attention = self._ddpm_caching_update(
                sequence, x, t, dt, p_x0=p_x0_cache, conf=confident_score,
                return_attention=return_attention)

            if (not torch.allclose(x_next, x)):
                p_x0_cache = None
            x = x_next

        if self.noise_removal: 
            t = min_t * torch.ones(bs, 1, device=self.device)
            unet_conditioning = self.noise(t)[0]
            padding_mask = (x != 0).float()
            x, attention = self.model(sequence, (x, None, None), unet_conditioning.squeeze(-1),
                                      padding_mask, return_attention=return_attention)
            x = x.argmax(dim=-1)
        return x, attention


class ProtobindMaskedDiffusion(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """The core Protobind-Diff model, which uses a Transformer decoder with RoPE.

    This model is designed for a masked diffusion task and supports conditioning
    on ESM-2 protein embeddings and generating ligands with a ChemformerTokenizer.
    """


    def __init__(self,
                 embedding_dim: int,
                 mask_index: int,
                 vocab_size: int,
                 decoder_params: Optional[dict] = None,
                 dropout: float = 0.2,
                 parametrization_strategy: str = 'subs',
                 **kwargs) -> None:
        """Initializes the ProtobindMaskedDiffusion model.

        Args:
            embedding_dim (int): The dimension of the protein sequence embeddings.
            mask_index (int): The token ID for the MASK token.
            vocab_size (int): The size of the ligand's vocabulary.
            decoder_params (Optional[dict]): A dictionary of parameters for the
                internal Transformer decoder (e.g., nhead, n_layers).
            dropout (float): The dropout rate.
            parametrization_strategy (str): The diffusion parameterization to use.
                Currently only 'subs' is supported.
        """
        super().__init__()

        self.neg_infinity = -1000000.0
        self.parametrization_strategy = parametrization_strategy
        self.decoder_name = decoder_params.pop('decoder_name')
        expand_feedforward = decoder_params.pop('expand_feedforward')
        self.mask_index = mask_index

        # Decoder options
        if self.decoder_name == 'decoder_re':
            self.decoder = Decoder_RoPE(vocab_size, embedding_dim, expand_feedforward=expand_feedforward,
                                        dropout=dropout, **decoder_params)
        else:
            raise ValueError(f"Model only supports decoder with rotary embeddings ('decoder_re'), got: {self.decoder_name}")

    def forward(self,
                sequence: Tuple[torch.Tensor, torch.Tensor],
                ligands: Tuple[torch.Tensor, torch.Tensor],
                sigma: torch.Tensor,
                mask_ligand: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """Performs the main forward pass of the diffusion model.

        Args:
            sequence (Tuple[torch.Tensor, torch.Tensor]): A tuple of the conditioning
                protein sequence embeddings and their lengths.
            ligands (Tuple[torch.Tensor, torch.Tensor]): A tuple
                containing the noised ligand `xt`and its length.
            sigma (torch.Tensor): The diffusion timestep (noise level).
            mask_ligand (torch.Tensor): The padding mask for the ligand.
            return_attention (bool): If True, returns attention weights from the decoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the final predicted logits
            and the attention weights.
        """

        sequence, sequence_lengths = sequence
        xt, ligand_lengths, _ = ligands

        # Decode ligand
        ligand_masked = xt.squeeze(-1).long()
        ligand_decoded, attention = self.decoder(ligand_masked,
                                                 sigma,
                                                 sequence,
                                                 sequence_lengths,
                                                 lig_padding_mask=None,
                                                 return_attention=return_attention)

        # Apply parametrization
        ligand_decoded = self.parametrization(ligand_decoded, xt)

        return ligand_decoded, attention

    def parametrization(self, logits, xt):
        """Applies the chosen parameterization to the model's output logits.

        The 'subs' strategy modifies the logits to represent the probability
        p(x_{t-1}|x_t), enforcing that unmasked tokens remain unchanged.

        Args:
            logits (torch.Tensor): The raw output logits from the decoder.
            xt (torch.Tensor): The noised input ligand at timestep t.

        Returns:
            torch.Tensor: The re-parameterized logits.
        """
        if self.parametrization_strategy == 'subs':
            # log prob at the mask index = - infinity
            logits[:, :, self.mask_index] += self.neg_infinity

            # Normalize the logits
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

            # Apply updates for unmasked tokens
            xt = xt.squeeze(-1)
            unmasked_indices = (xt != self.mask_index)
            logits[unmasked_indices] = self.neg_infinity
            logits[unmasked_indices, xt[unmasked_indices].long()] = 0
        else:
            raise NotImplementedError(f'Parametrization strategy {self.parametrization_strategy} not implemented')
        return logits