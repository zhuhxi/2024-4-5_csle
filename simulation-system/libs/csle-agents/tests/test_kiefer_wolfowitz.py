import numpy as np
import pytest
import pytest_mock
import csle_common.constants.constants as constants
from csle_common.dao.training.experiment_config import ExperimentConfig
from csle_common.dao.training.agent_type import AgentType
from csle_common.dao.training.hparam import HParam
from csle_common.dao.training.player_type import PlayerType
from csle_common.dao.training.policy_type import PolicyType
from csle_agents.agents.kiefer_wolfowitz.kiefer_wolfowitz_agent import KieferWolfowitzAgent
import csle_agents.constants.constants as agents_constants
from gym_csle_stopping_game.dao.stopping_game_config import StoppingGameConfig
from gym_csle_stopping_game.dao.stopping_game_defender_pomdp_config import StoppingGameDefenderPomdpConfig
from gym_csle_stopping_game.util.stopping_game_util import StoppingGameUtil
from csle_common.dao.training.random_policy import RandomPolicy
import gym_csle_stopping_game.constants.constants as env_constants


class TestKieferWolfowitzSuite(object):
    """
    Test suite for the KieferWolfoWitzAgent
    """

    @pytest.fixture
    def experiment_config(self) -> ExperimentConfig:
        """
        Fixture, which is run before every test. It sets up an example experiment config

        :return: the example experiment config
        """
        experiment_config = ExperimentConfig(
            output_dir=f"{constants.LOGGING.DEFAULT_LOG_DIR}kiefer_wolfowitz_test", title="Kiefer-Wolfowitz test",
            random_seeds=[399],
            agent_type=AgentType.KIEFER_WOLFOWITZ,
            log_every=1,
            hparams={
                agents_constants.KIEFER_WOLFOWITZ.N: HParam(value=2, name=constants.T_SPSA.N,
                                                            descr="the number of training iterations"),
                agents_constants.KIEFER_WOLFOWITZ.L: HParam(value=1, name=agents_constants.KIEFER_WOLFOWITZ.L,
                                                            descr="the number of stop actions"),
                agents_constants.KIEFER_WOLFOWITZ.DELTA: HParam(value=2, name=agents_constants.KIEFER_WOLFOWITZ.DELTA,
                                                                descr="perturbation size"),
                agents_constants.KIEFER_WOLFOWITZ.INITIAL_ALPHA: HParam(
                    value=10, name=agents_constants.KIEFER_WOLFOWITZ.INITIAL_ALPHA, descr="initial step size"),
                agents_constants.COMMON.EVAL_BATCH_SIZE: HParam(value=10, name=agents_constants.COMMON.EVAL_BATCH_SIZE,
                                                                descr="number of iterations to evaluate theta"),
                agents_constants.KIEFER_WOLFOWITZ.THETA1: HParam(value=[-3, -3, -3],
                                                                 name=agents_constants.KIEFER_WOLFOWITZ.THETA1,
                                                                 descr="initial thresholds"),
                agents_constants.COMMON.SAVE_EVERY: HParam(value=1000, name=agents_constants.COMMON.SAVE_EVERY,
                                                           descr="how frequently to save the model"),
                constants.T_SPSA.GRADIENT_BATCH_SIZE: HParam(
                    value=1, name=constants.T_SPSA.GRADIENT_BATCH_SIZE,
                    descr="the batch size of the gradient estimator"),
                agents_constants.COMMON.CONFIDENCE_INTERVAL: HParam(
                    value=0.95, name=agents_constants.COMMON.CONFIDENCE_INTERVAL,
                    descr="confidence interval"),
                agents_constants.COMMON.MAX_ENV_STEPS: HParam(
                    value=500, name=agents_constants.COMMON.MAX_ENV_STEPS,
                    descr="maximum number of steps in the environment (for envs with infinite horizon generally)"),
                agents_constants.COMMON.RUNNING_AVERAGE: HParam(
                    value=100, name=agents_constants.COMMON.RUNNING_AVERAGE,
                    descr="the number of samples to include when computing the running avg"),
                agents_constants.COMMON.GAMMA: HParam(
                    value=0.99, name=agents_constants.COMMON.GAMMA,
                    descr="the discount factor"),
                agents_constants.KIEFER_WOLFOWITZ.POLICY_TYPE: HParam(
                    value=PolicyType.MULTI_THRESHOLD, name=agents_constants.KIEFER_WOLFOWITZ.POLICY_TYPE,
                    descr="policy type for the execution")
            },
            player_type=PlayerType.DEFENDER, player_idx=0
        )
        return experiment_config

    @pytest.fixture
    def pomdp_config(self) -> StoppingGameDefenderPomdpConfig:
        """
        Fixture, which is run before every test. It sets up an input POMDP config

        :return: The example config
        """
        L = 1
        R_INT = -5
        R_COST = -5
        R_SLA = 1
        R_ST = 5
        p = 0.1
        n = 100

        attacker_stage_strategy = np.zeros((3, 2))
        attacker_stage_strategy[0][0] = 0.9
        attacker_stage_strategy[0][1] = 0.1
        attacker_stage_strategy[1][0] = 0.9
        attacker_stage_strategy[1][1] = 0.1
        attacker_stage_strategy[2] = attacker_stage_strategy[1]

        stopping_game_config = StoppingGameConfig(
            A1=StoppingGameUtil.attacker_actions(), A2=StoppingGameUtil.defender_actions(), L=L, R_INT=R_INT,
            R_COST=R_COST,
            R_SLA=R_SLA, R_ST=R_ST, b1=np.array(list(StoppingGameUtil.b1())),
            save_dir="./results",
            T=StoppingGameUtil.transition_tensor(L=L, p=p),
            O=StoppingGameUtil.observation_space(n=n),
            Z=StoppingGameUtil.observation_tensor(n=n),
            R=StoppingGameUtil.reward_tensor(R_SLA=R_SLA, R_INT=R_INT, R_COST=R_COST, L=L, R_ST=R_ST),
            S=StoppingGameUtil.state_space(), env_name="csle-stopping-game-v1", checkpoint_traces_freq=100000,
            gamma=1)
        pomdp_config = StoppingGameDefenderPomdpConfig(
            stopping_game_config=stopping_game_config, stopping_game_name="csle-stopping-game-v1",
            attacker_strategy=RandomPolicy(actions=list(stopping_game_config.A2),
                                           player_type=PlayerType.ATTACKER,
                                           stage_policy_tensor=list(attacker_stage_strategy)),
            env_name="csle-stopping-game-pomdp-defender-v1")
        return pomdp_config

    def test_create_agent(self, mocker: pytest_mock.MockFixture, experiment_config: ExperimentConfig) -> None:
        """
        Tests creation of the KieferWolfowitzAgent

        :return: None
        """
        emulation_env_config = mocker.MagicMock()
        simulation_env_config = mocker.MagicMock()
        KieferWolfowitzAgent(emulation_env_config=emulation_env_config, simulation_env_config=simulation_env_config,
                             experiment_config=experiment_config)

    def test_run_agent(self, mocker: pytest_mock.MockFixture, experiment_config: ExperimentConfig,
                       pomdp_config: StoppingGameDefenderPomdpConfig) -> None:
        """
        Tests running the agent

        :param mocker: object for mocking API calls
        :param experiment_config: the example experiment config
        :param pomdp_config: the example POMDP config

        :return: None
        """
        # Mock emulation and simulation configs
        emulation_env_config = mocker.MagicMock()
        simulation_env_config = mocker.MagicMock()

        # Set attributes of the mocks
        simulation_env_config.configure_mock(**{
            "name": "simulation-test-env", "gym_env_name": "csle-stopping-game-pomdp-defender-v1",
            "simulation_env_input_config": pomdp_config
        })
        emulation_env_config.configure_mock(**{
            "name": "emulation-test-env"
        })

        # Mock metastore facade
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.save_training_job', return_value=True)
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.save_experiment_execution',
                     return_value=True)
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.update_training_job', return_value=True)
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.update_experiment_execution',
                     return_value=True)
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.save_simulation_trace', return_value=True)
        mocker.patch('csle_common.metastore.metastore_facade.MetastoreFacade.save_multi_threshold_stopping_policy',
                     return_value=True)
        agent = KieferWolfowitzAgent(emulation_env_config=emulation_env_config,
                                     simulation_env_config=simulation_env_config,
                                     experiment_config=experiment_config)
        experiment_execution = agent.train()
        assert experiment_execution is not None
        assert experiment_execution.descr != ""
        assert experiment_execution.id is not None
        assert experiment_execution.config == experiment_config
        assert agents_constants.COMMON.AVERAGE_RETURN in experiment_execution.result.plot_metrics
        assert agents_constants.COMMON.RUNNING_AVERAGE_RETURN in experiment_execution.result.plot_metrics
        assert env_constants.ENV_METRICS.INTRUSION_LENGTH in experiment_execution.result.plot_metrics
        assert agents_constants.COMMON.RUNNING_AVERAGE_INTRUSION_LENGTH in experiment_execution.result.plot_metrics
        assert env_constants.ENV_METRICS.INTRUSION_START in experiment_execution.result.plot_metrics
        assert agents_constants.COMMON.RUNNING_AVERAGE_INTRUSION_START in experiment_execution.result.plot_metrics

        assert env_constants.ENV_METRICS.TIME_HORIZON in experiment_execution.result.plot_metrics
        assert agents_constants.COMMON.RUNNING_AVERAGE_TIME_HORIZON in experiment_execution.result.plot_metrics
        assert env_constants.ENV_METRICS.AVERAGE_UPPER_BOUND_RETURN in experiment_execution.result.plot_metrics
        assert env_constants.ENV_METRICS.AVERAGE_DEFENDER_BASELINE_STOP_ON_FIRST_ALERT_RETURN in \
               experiment_execution.result.plot_metrics
        for seed in experiment_config.random_seeds:
            assert seed in experiment_execution.result.all_metrics
            assert agents_constants.KIEFER_WOLFOWITZ.THETAS in experiment_execution.result.all_metrics[seed]
            assert agents_constants.COMMON.AVERAGE_RETURN in experiment_execution.result.all_metrics[seed]
            assert agents_constants.COMMON.RUNNING_AVERAGE_RETURN in experiment_execution.result.all_metrics[seed]
            assert agents_constants.KIEFER_WOLFOWITZ.THRESHOLDS in experiment_execution.result.all_metrics[seed]
            assert (agents_constants.COMMON.RUNNING_AVERAGE_INTRUSION_START in
                    experiment_execution.result.all_metrics[seed])
            assert (agents_constants.COMMON.RUNNING_AVERAGE_TIME_HORIZON in
                    experiment_execution.result.all_metrics[seed])
            assert (agents_constants.COMMON.RUNNING_AVERAGE_INTRUSION_LENGTH in
                    experiment_execution.result.all_metrics[seed])
            assert env_constants.ENV_METRICS.INTRUSION_START in experiment_execution.result.all_metrics[seed]
            assert env_constants.ENV_METRICS.TIME_HORIZON in experiment_execution.result.all_metrics[seed]
            assert env_constants.ENV_METRICS.AVERAGE_UPPER_BOUND_RETURN in experiment_execution.result.all_metrics[seed]
            assert (env_constants.ENV_METRICS.AVERAGE_DEFENDER_BASELINE_STOP_ON_FIRST_ALERT_RETURN in
                    experiment_execution.result.all_metrics[seed])
