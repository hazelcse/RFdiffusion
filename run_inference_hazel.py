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


# deterministic seed setting
def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# main function for running protein generation
def diffusion_process(conf: OmegaConf) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # initialize sampler and target/contig, responsible for generating protein structures
    sampler = iu.sampler_selector(conf)

    # protein generation starts here
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    # iterate over number of designs it needs to generate
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        # each design generation FUNCTION
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue
        
        # initialise protein structure and stacks for storing INTERMEDIATE RESULTS
        x_init, seq_init = sampler.sample_init() # initial coordinates and sequence
        denoised_xyz_stack = [] # stacks for storing denoised coordinates
        px0_xyz_stack = [] # stacks for storing predicted coordinates
        seq_stack = [] # stacks for storing sequences ARE THESE THE INTERMEDIATE SEQUENCES?
        plddt_stack = [] # stacks for scoring pLDDT scores NECESSARY FOR EVALUATION?

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)

        # loop over number of reverse diffusion time steps
        # actual diffusion process i guess
        for num,t in enumerate(range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1)):
            
            # single diffusion step
            # each TIMESTEP FUNCTION
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )

            # appends results to stacks
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

            # save the intermediate results
            if conf.inference.dump_pdb:
                bfacts = plddt.cpu().numpy()[0]
                protein = px0.cpu().numpy()[:,:4]
                pdb_str,line_num = "",0
                for n,residue in enumerate(protein):
                    for xyz,atom in zip(residue,[" N  ", " CA ", " C  ", " O  "]):
                        pdb_str += "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % ("ATOM",line_num,atom,"GLY","A",n,*xyz,1,bfacts[n])
                        line_num += 1
                dump_pdb_path = os.path.join(conf.inference.dump_pdb_path,f"{num}.pdb")
                with open(dump_pdb_path,"w") as handle:
                    handle.write(pdb_str + "TER")


        # flip order for better visualization in pymol
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

        # for logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)
        bfact_stack = torch.flip(
            plddt_stack,
            [
                0,
            ],
        )

        # Save outputs
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
            time=time.time() - start_time,
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

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")