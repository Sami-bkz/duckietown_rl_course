# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

# from stable_baselines3.common.atari_wrappers import (  # isort:skip
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )

from duckiesim.rl.process_data_with_reward import process_dataset
from duckietownrl.gym_duckietown import envs
from duckiesim.rl.rl_utils import evaluate_ppo

# from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DuckieTownRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "DuckietownDiscrete-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.05
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.conv1 = nn.Conv2d(envs.single_observation_space.shape[-1], 32, 4, stride=2)  # Modification : 3 canaux pour RGB
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        
        self.flatten_size = self._get_flatten_size(envs.single_observation_space.shape)
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(envs.single_observation_space.shape[-1], 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            layer_init(nn.Linear(self.flatten_size, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
    

    def get_value(self, x):
        return self.critic(self.network(x.to(dtype=torch.float32).permute(0, 3, 1, 2)/255.0))
    
    def get_policy(self, x):
        hidden = self.network(x.to(dtype=torch.float32).permute(0, 3, 1, 2)/255.0)
        logits = self.actor(hidden)
        return nn.functional.softmax(logits, dim=1)
    
    def _get_flatten_size(self, input_shape):
        """
        Compute the size of the output of the network, given the input shape
        """
        dummy_input = torch.zeros(1, *input_shape)  # Batch size = 1
        x = self.conv1(dummy_input.permute(0, 3, 1, 2))  # ReLU appliqué directement
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()  # Taille totale de la sortie aplatie


    def get_policy(self, x): 
        hidden = self.network(x.to(dtype=torch.float32).permute(0, 3, 1, 2)/255.0)
        logits = self.actor(hidden)
        # softmax 
        return nn.functional.softmax(logits, dim=1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.to(dtype=torch.float32).permute(0, 3, 1, 2)/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
   # writer = SummaryWriter(f"runs/{run_name}")
   # writer.add_text(
   #     "hyperparameters",
   #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
   # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load("/home/duckietown_rl_course/duckiereal/imitation_learning/models_imitation/behavioral_cloning_ppo.pt"))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # MODEL PATH 
    # Vérifie si le répertoire 'model' existe, sinon le crée
    if not os.path.exists('model'):
        os.makedirs('model')
    # Récupère la liste des expériences existantes
    exps = [d for d in os.listdir('model') if d.startswith('exp_')]
    # Si aucun dossier exp_ n'existe, on commence à exp_0
    if not exps:
        run_name = 'exp_0'
    else:
        # Sinon, on récupère le numéro de l'expérience la plus élevée
        last_exp_num = max(int(d.split('_')[1]) for d in exps)
        # On incrémente le numéro de l'expérience pour obtenir le nouveau nom
        run_name = f'exp_{last_exp_num + 1}'

    # Utilise le nom d'expérience pour sauvegarder le modèle
    model_path = f"model/{run_name}"
    # Crée le répertoire si il n'existe pas
    os.makedirs(model_path, exist_ok=True)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    best_episodic_return = -1e9
    episodic_return = 0.0
    best_eval_score = -1e9
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration {iteration}/{args.num_iterations}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        print("\tSampling in progress...")
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(np.array(action.cpu().tolist()))
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]  # Mise à jour ici
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({"charts/episodic_return": info["episode"]["r"], "charts/episodic_length": info["episode"]["l"]}, global_step) if args.track else None
                        
        print("\tEvaluation in progress...")
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if b_returns.mean() > best_episodic_return and iteration >= 20:
            best_episodic_return = b_returns.mean()
            eval_score = evaluate_ppo(agent, args.env_id, seed=args.seed)
            print(f"Eval score: {eval_score}, Best eval score : {best_eval_score}")
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(agent.state_dict(), model_path+f"/{args.exp_name}_{iteration}_{eval_score}.pt")
                print(f"model saved to {model_path}")

            print(f"Step: {global_step}, Eval score: {eval_score}")

        print("\tOptimization step in progress...")
        # Optimizing the policy and value network
        b_inds = torch.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # mb_inds = torch.tensor(mb_inds).to(device)
               # print(mb_inds.dtype)
               # print(b_obs.dtype)
                # print(b_actions.dtype)
                # print(b_actions.long().dtype)
                # print('lin1 : ', b_obs[mb_inds].shape)
                # print('lin2 : ', b_actions.long()[mb_inds].shape)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                if iteration <= 20:
                    loss = v_loss * args.vf_coef
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                print(loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
       # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
       # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
       # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
       # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
       # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
       # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
       # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
       # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
       # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.track : 
            wandb.log({"charts/learning_rate": optimizer.param_groups[0]["lr"], "charts/SPS": int(global_step / (time.time() - start_time))}, global_step)
            wandb.log({"losses/value_loss": v_loss.item(), "losses/policy_loss": pg_loss.item(), "losses/entropy": entropy_loss.item(), "losses/old_approx_kl": old_approx_kl.item(), "losses/approx_kl": approx_kl.item(), "losses/clipfrac": np.mean(clipfracs)}, global_step)

        # On évalue la politique
       # if episodic_return > best_episodic_return or global_step % 10_000 == 0:
       #     best_episodic_return = episodic_return
       #     eval_score = evaluate_ppo(agent, args.env_id, seed=args.seed, tau=args.tau_soft)
       #     if eval_score > best_eval_score:
       #         best_eval_score = eval_score
       #         if args.save_model:
       #             torch.save(agent.state_dict(), model_path+f"/{args.exp_name}_{global_step}_{eval_score}.pt")
       #             print(f"model saved to {model_path}")

       #     print(f"Step: {global_step}, Eval score: {eval_score}")

    envs.close()