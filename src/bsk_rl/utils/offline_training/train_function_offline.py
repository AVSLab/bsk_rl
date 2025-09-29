import glob
import os
import pathlib
import shutil
from pathlib import Path
import torch

torch.set_num_threads(11)
os.environ["MKL_NUM_THREADS"] = "11"
import ray
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.core import DEFAULT_MODULE_ID
from typing import Dict
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
import torch
import tempfile

class BCActor(torch.nn.Module):
    """A wrapper for the encoder and policy networks of a PPORLModule.

    Args:
        encoder_network: The encoder network of the PPORLModule.
        policy_network: The policy network of the PPORLModule.
        distribution_cls: The distribution class to construct with the logits outputed
            by the policy network.
    """

    def __init__(
        self,
        encoder_network: torch.nn.Module,
        policy_network: torch.nn.Module,
        distribution_cls: torch.distributions.Distribution,
    ):
        super().__init__()
        self.encoder_network = encoder_network
        self.policy_network = policy_network
        self.distribution_cls = distribution_cls

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.distributions.Distribution:
        """Return an action distribution output by the policy network.

        batch: A dict containing the key "obs" mapping to a torch tensor of
            observations.

        """
        # The encoder network has outputs for the actor and critic heads of the
        # PPORLModule. We only want the outputs for the actor head.
        encoder_out = self.encoder_network(batch)[ENCODER_OUT]#[ACTOR]
        action_logits = self.policy_network(encoder_out)
        distribution = self.distribution_cls(logits=action_logits)
        return distribution
    

class BCCritic(torch.nn.Module):
    """A wrapper for the encoder and policy networks of a PPORLModule.

    Args:
        encoder_network: The encoder network of the PPORLModule.
        policy_network: The policy network of the PPORLModule.
        distribution_cls: The distribution class to construct with the logits outputed
            by the policy network.
    """

    def __init__(
        self,
        encoder_network: torch.nn.Module,
        critic_network: torch.nn.Module,
    ):
        super().__init__()
        self.encoder_network = encoder_network
        self.critic_network = critic_network

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.distributions.Distribution:
        """Return an action distribution output by the policy network.

        batch: A dict containing the key "obs" mapping to a torch tensor of
            observations.

        """
        # The encoder network has outputs for the actor and critic heads of the
        # PPORLModule. We only want the outputs for the actor head.
        encoder_out = self.encoder_network(batch)[ENCODER_OUT]#[ACTOR]
        critic = self.critic_network(encoder_out)
        return critic


def train_ppo_module_with_bc_finetune(
    dataset: ray.data.Dataset, module: RLModule, checkpoint_dir = pathlib.Path().absolute(), update_critic = False
) -> str:
    """Train an Actor with BC finetuning on dataset.

    Args:
        dataset: The dataset to train on.
        module_spec: The module spec of the PPORLModule that will be trained
            after its encoder and policy networks are pretrained with BC.

    Returns:
        The path to the checkpoint of the pretrained PPORLModule.
    """
    batch_size = 512
    learning_rate = 1e-3
    num_epochs = 10

    # We want to pretrain the encoder and policy networks of the RLModule. We don't want
    # to pretrain the value network. The actor will use the Categorical distribution,
    # as its output distribution since we are training on the CartPole environment which
    # has a discrete action space.
    BCActorNetwork = BCActor(module.encoder.actor_encoder, module.pi, torch.distributions.Categorical)
    optim = torch.optim.Adam(BCActorNetwork.parameters(), lr=learning_rate)
    if update_critic:
        BCCriticNetwork = BCCritic(module.encoder.critic_encoder, module.vf)
        optim_critic = torch.optim.Adam(BCCriticNetwork.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in dataset.iter_torch_batches(
            batch_size=batch_size, dtypes=torch.float32
        ):
            action_dist = BCActorNetwork(batch)
            loss = -torch.mean(action_dist.log_prob(batch["actions"]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if update_critic:
                critic_loss = torch.mean((batch["rewards"] - BCCriticNetwork(batch).squeeze())**2)
                optim_critic.zero_grad()
                critic_loss.backward()
                optim_critic.step()
        
        print(f"Epoch {epoch} loss: {loss.detach().item()}")
        if update_critic:
            print(f"Epoch {epoch} critic loss: {critic_loss.detach().item()}")

    if checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp()
    module.save_to_path(checkpoint_dir / "learner_group" / "learner" /  "rl_module" /  DEFAULT_MODULE_ID)
    return checkpoint_dir


def initialize_CL_callbacks(total_timesteps):
    class CLCallbacks(WrappedEpisodeDataCallbacks):

        def on_episode_start(
            self,
            *,
            metrics_logger=None,
            env=None,
            env_index,
            **kwargs,
        ) -> None:

            try:
                n_steps = metrics_logger.peek("num_env_steps_sampled_lifetime")
                if n_steps is None:
                    task = 0.0
                else:
                    task = n_steps / total_timesteps
            except KeyError:
                task = 0.0

            env.envs[env_index].set_task(task)

    return CLCallbacks


def train_model_offline(
    model_name,
    output_directory,
    env,
    offline_data,
    env_args={},
    n_envs=1,
    checkpoint_frequency=1,
    checkpoints_to_keep=2,
    reload_frequency=500_000,
    total_timesteps=1_000_000,
    training_args={},
    rl_module_args={},
    temp_dir="/tmp",
    n_steps_switch=None,
    continue_previous=False,
    CL_mode=False,
    N_CORES=None,
    sMDP_mode=True,
    connector_modules={},
    # slew_server_mode=False,
):

    # if slew_server_mode:
    #     print("Starting SLEW SERVER")
    #     slew_server = SlewServer()
    #     slew_server.start_slew_server()

    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    start_from_zero = False

    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs + 1,
            sample_timeout_s=1000.0,
            **connector_modules,
        )
        .environment(
            env=env,
            env_config=env_args,
        )
        .callbacks(
            WrappedEpisodeDataCallbacks
            if not CL_mode
            else initialize_CL_callbacks(total_timesteps)
        )
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            metrics_episode_collection_timeout_s=180,
        )
        .checkpointing(export_native_model_files=True)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(num_gpus=0)
    )
    if sMDP_mode:
        config = config.training(
            **training_args,
            learner_class=TimeDiscountedGAEPPOTorchLearner,
        )
    else:
        config = config.training(**training_args)

    config.rl_module(**rl_module_args)
    config.logger_config = dict(
        type=UnifiedLogger, logdir=output_directory / model_name
    )
    # config.debugging(log_level="DEBUG")

    if continue_previous:
        checkpoint_path_dir = (
            output_directory / model_name
        )
        checkpoints = glob.glob(str(checkpoint_path_dir) + "/checkpoint_*")
        if len(checkpoints) == 0:
            print("No model to re-load and continue training")
            start_from_zero = True

        if not start_from_zero:
            checkpoint_number_str = ""
            checkpoint_number_int = 0
            # Figure out the most recent checkpoint
            for checkpoint_number in checkpoints:
                checkpoint_folder_i = checkpoint_number.split("/")[-1]
                checkpoint_number_i = int(checkpoint_folder_i.split("_")[-1])
                if checkpoint_number_i > checkpoint_number_int:
                    checkpoint_number_int = checkpoint_number_i
                    checkpoint_number_str = checkpoint_folder_i.split("_")[-1]

            latest_checkpoint_dir = checkpoint_path_dir / f"checkpoint_{checkpoint_number_str}"
            if not os.path.exists(output_directory):
                print(f"Selected path does not exist: {latest_checkpoint_dir}")
                exit()

    if not continue_previous or start_from_zero:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=N_CORES,
            object_store_memory=2_000_000_000,  # 2 GB
            _temp_dir=temp_dir,
        )
        
        ppo = PPO(config)
        iter = 0
        step = 0
        if n_steps_switch is not None:
            cond_switch = False
            checkpoint_saved = False
        else:
            cond_switch = True
            checkpoint_saved = True

        while True:
            prev_step = step
            results = ppo.train()
            step = results["num_env_steps_trained_lifetime"]

            checkpoint_path = (
                output_directory / model_name / f"checkpoint_{str(iter).zfill(6)}"
            )
            if iter % checkpoint_frequency == 0:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)

            if step > total_timesteps:
                break

            if cond_switch is False and (step + reload_frequency) > n_steps_switch:
                cond_switch = True

            if step % reload_frequency < prev_step % reload_frequency:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                if cond_switch and checkpoint_saved is False:
                    checkpoint_saved = True
                    shutil.copytree(output_directory / model_name, output_directory / f"{model_name}_switch")
                ray.shutdown()

                # if slew_server_mode:
                #     slew_server.stop_slew_server()
                #     slew_server = SlewServer()
                #     slew_server.start_slew_server()

                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=N_CORES,
                    object_store_memory=2_000_000_000,  # 2 GB
                    _temp_dir=temp_dir,
                )
                ppo = PPO.from_checkpoint(checkpoint_path)

            if iter > checkpoints_to_keep * checkpoint_frequency - 1:
                for i in range(checkpoint_frequency):
                    remove_dir = (
                        output_directory
                        / model_name
                        / f"checkpoint_{str(iter - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                    )
                    try:
                        shutil.rmtree(remove_dir)
                    except FileNotFoundError:
                        pass
                break

            iter += 1

        ray.shutdown()

        latest_checkpoint_dir = checkpoint_path   

    module = RLModule.from_checkpoint(
        latest_checkpoint_dir /  "learner_group" / "learner" /  "rl_module" /  DEFAULT_MODULE_ID,
    )

    train_ppo_module_with_bc_finetune(
        offline_data,
        module,
        latest_checkpoint_dir,
    )

    
