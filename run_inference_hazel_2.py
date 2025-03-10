# imports
import re
import os, time, pickle
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from util import writepdb_multi, writepdb
from inference import utils as iu
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob

# deterministic seed setting function
def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def initialise_diffusion(conf: OmegaConf) -> None:
    if conf.inference.deterministic:
        make_deterministic()

    # initialize sampler and target/contig, responsible for generating protein structures
    sampler = iu.sampler_selector(conf)

    # numbering for output pdb
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(r".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1
    return design_startnum, sampler

def cleanup_stacks(denoised_xyz_stack, px0_xyz_stack, plddt_stack):
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(
        denoised_xyz_stack,
        [
            0,
        ],
    )
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(
        px0_xyz_stack,
        [
            0,
        ],
    )
    plddt_stack = torch.stack(plddt_stack)
    bfact_stack = torch.flip(
        plddt_stack,
        [
            0,
        ],
    )
    return denoised_xyz_stack, px0_xyz_stack, plddt_stack, bfact_stack

def wrapup_diffusion(out_prefix, sampler, seq_stack, seq_init, denoised_xyz_stack, bfact_stack, plddt_stack, px0_xyz_stack):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    final_seq = seq_stack[-1]

    # Output glycines, except for motif region
    final_seq = torch.where(
        torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
    )  # 7 is glycine

    bfacts = torch.ones_like(final_seq.squeeze())
    bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
    # pX0 last step
    out = f"{out_prefix}.pdb"

    # don't output sidechains
    writepdb(
        out,
        denoised_xyz_stack[0, :, :4],
        final_seq,
        sampler.binderlen,
        chain_idx=sampler.chain_idx,
        bfacts=bfact_stack[0],
    )

    # run metadata
    trb = dict(
        config=OmegaConf.to_container(sampler._conf, resolve=True),
        plddt=plddt_stack.cpu().numpy(),
        device=torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "CPU",
    )
    if hasattr(sampler, "contig_map"):
        for key, value in sampler.contig_map.get_mappings().items():
            trb[key] = value
    with open(f"{out_prefix}.trb", "wb") as f_out:
        pickle.dump(trb, f_out)

    if sampler.inf_conf.write_trajectory:
        # trajectory pdbs
        traj_prefix = (
            os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
        )
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        out = f"{traj_prefix}_Xt-1_traj.pdb"
        writepdb_multi(
            out,
            denoised_xyz_stack,
            bfact_stack,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
            chain_ids=sampler.chain_idx,
        )

        out = f"{traj_prefix}_pX0_traj.pdb"
        writepdb_multi(
            out,
            px0_xyz_stack,
            bfact_stack,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
            chain_ids=sampler.chain_idx,
        )