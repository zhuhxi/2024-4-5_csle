# Imports
import csle_common.constants.constants as constants
from csle_common.dao.training.experiment_config import ExperimentConfig
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.dao.training.agent_type import AgentType
from csle_common.dao.training.hparam import HParam
from csle_common.dao.training.player_type import PlayerType
from csle_agents.agents.ppo.ppo_agent import PPOAgent
import csle_agents.constants.constants as agents_constants
from csle_common.dao.training.tabular_policy import TabularPolicy

# Select emulation configuration from the metastore
emulation_env_config = MetastoreFacade.get_emulation_by_name("csle-level9-010")

# Select simulation configuration from the metastore
simulation_env_config = MetastoreFacade.get_simulation_by_name(
                                    "csle-stopping-pomdp-defender-010")

# Setup the reinforcement learning experiment
experiment_config = ExperimentConfig(
                    output_dir=f"{constants.LOGGING.DEFAULT_LOG_DIR}ppo_test",
                    title="PPO test",
                    random_seeds=[399, 98912, 999], agent_type=AgentType.PPO,
                    log_every=1, hparams={..},
                    player_type=PlayerType.DEFENDER, player_idx=0)
agent = PPOAgent(emulation_env_config=emulation_env_config,
simulation_env_config=simulation_env_config,
experiment_config=experiment_config)

# Run the PPO algorithm to learn defender policies
experiment_execution = agent.train()

# Save the experiment results and the learned policies
MetastoreFacade.save_experiment_execution(experiment_execution)
for policy in experiment_execution.result.policies.values():
    MetastoreFacade.save_ppo_policy(ppo_policy=policy)