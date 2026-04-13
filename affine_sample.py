from tqdm import trange
import torch

NEG_INF = -torch.inf

def _sanitize_logp(logp):
    return torch.where(torch.isfinite(logp), logp, torch.full_like(logp, NEG_INF))

def mask_prior_and_summaries(theta, summs, low, high):
    mask = ((theta >= low) & (theta <= high)).all(dim=-1)
    return theta.float()[mask], summs.float()[mask], mask

def mask_prior(theta, low, high):
    mask = ((theta >= low) & (theta <= high)).all(dim=-1)
    return theta.float()[mask], mask

def affine_sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2, progress_bar=True):
    current_state1 = torch.as_tensor(walkers1)
    current_state2 = torch.as_tensor(walkers2)

    chain = [torch.cat([current_state1, current_state2], dim=0).clone()]

    iterator = trange(n_steps - 1) if progress_bar else range(n_steps - 1)

    with torch.no_grad():
        logp_current1 = _sanitize_logp(log_prob(current_state1))
        logp_current2 = _sanitize_logp(log_prob(current_state2))

        for _ in iterator:
            partners1 = current_state2[torch.randint(0, n_walkers, (n_walkers,), device=current_state2.device)]
            z1 = 0.5 * (1.0 + torch.rand(n_walkers, device=current_state1.device))**2
            proposed_state1 = partners1 + z1[:, None] * (current_state1 - partners1)

            logp_proposed1 = _sanitize_logp(log_prob(proposed_state1))
            log_alpha1 = (n_params - 1) * torch.log(z1) + (logp_proposed1 - logp_current1)
            accept1 = torch.log(torch.rand(n_walkers, device=current_state1.device)) < log_alpha1

            current_state1 = torch.where(accept1[:, None], proposed_state1, current_state1)
            logp_current1 = torch.where(accept1, logp_proposed1, logp_current1)

            partners2 = current_state1[torch.randint(0, n_walkers, (n_walkers,), device=current_state1.device)]
            z2 = 0.5 * (1.0 + torch.rand(n_walkers, device=current_state2.device))**2
            proposed_state2 = partners2 + z2[:, None] * (current_state2 - partners2)

            logp_proposed2 = _sanitize_logp(log_prob(proposed_state2))
            log_alpha2 = (n_params - 1) * torch.log(z2) + (logp_proposed2 - logp_current2)
            accept2 = torch.log(torch.rand(n_walkers, device=current_state2.device)) < log_alpha2

            current_state2 = torch.where(accept2[:, None], proposed_state2, current_state2)
            logp_current2 = torch.where(accept2, logp_proposed2, logp_current2)

            chain.append(torch.cat([current_state1, current_state2], dim=0).clone())

    return torch.stack(chain, dim=0)