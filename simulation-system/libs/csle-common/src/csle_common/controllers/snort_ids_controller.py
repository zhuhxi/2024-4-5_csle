import logging
from typing import List
import grpc
import time
from csle_common.dao.emulation_config.emulation_env_config import EmulationEnvConfig
from csle_common.dao.emulation_config.snort_managers_info import SnortIdsManagersInfo
import csle_common.constants.constants as constants
import csle_collector.constants.constants as csle_collector_constants
import csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc
import csle_collector.snort_ids_manager.snort_ids_manager_pb2
import csle_collector.snort_ids_manager.query_snort_ids_manager
import csle_collector.snort_ids_manager.snort_ids_manager_util
from csle_common.util.emulation_util import EmulationUtil


class SnortIDSController:
    """
    Class managing operations related to Snort IDS Managers
    """

    @staticmethod
    def start_snort_idses(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                          logger: logging.Logger) -> None:
        """
        Utility function for starting the Snort IDSes

        :param emulation_env_config: the emulation env configuration
        :param physical_server_ip: the ip of the phsyical server
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    logger.info(f"Starting the Snort IDS on IP: {c.docker_gw_bridge_ip}")
                    SnortIDSController.start_snort_ids(emulation_env_config=emulation_env_config,
                                                       ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def stop_snort_idses(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                         logger: logging.Logger) -> None:
        """
        Utility function for stopping the Snort IDSes

        :param emulation_env_config: the emulation env configuration
        :param physical_server_ip: the ip of the physical server
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    SnortIDSController.stop_snort_ids(emulation_env_config=emulation_env_config,
                                                      ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def start_snort_ids(emulation_env_config: EmulationEnvConfig, ip: str, logger: logging.Logger) -> None:
        """
        Utility function for starting the Snort IDS on a specific IP

        :param emulation_env_config: the emulation env configuration
        :param ip: the ip of the container
        :param logger: the logger to use for logging
        :return: None
        """
        SnortIDSController.start_snort_manager(emulation_env_config=emulation_env_config, ip=ip, logger=logger)
        ids_monitor_dto = SnortIDSController.get_snort_idses_monitor_threads_statuses_by_ip_and_port(
            port=emulation_env_config.snort_ids_manager_config.snort_ids_manager_port, ip=ip)
        if not ids_monitor_dto.snort_ids_running:
            logger.info(f"Snort IDS is not running on {ip}, starting it.")
            subnetmask = f"{emulation_env_config.execution_id}.{emulation_env_config.level}" \
                         f"{constants.CSLE.CSLE_LEVEL_SUBNETMASK_SUFFIX}"
            ingress_interface = constants.NETWORKING.ETH2
            egress_interface = constants.NETWORKING.ETH0
            logger.info(f"Subnetmask: {subnetmask}, ingress interface: {ingress_interface}, "
                        f"egress interface: {egress_interface}")
            # Open a gRPC session
            with grpc.insecure_channel(
                    f'{ip}:'
                    f'{emulation_env_config.snort_ids_manager_config.snort_ids_manager_port}',
                    options=constants.GRPC_SERVERS.GRPC_OPTIONS) as channel:
                stub = csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc.SnortIdsManagerStub(channel)
                csle_collector.snort_ids_manager.query_snort_ids_manager.start_snort_ids(
                    stub=stub, ingress_interface=ingress_interface, egress_interface=egress_interface,
                    subnetmask=subnetmask)

    @staticmethod
    def stop_snort_ids(emulation_env_config: EmulationEnvConfig, ip: str, logger: logging.Logger) -> None:
        """
        Utility function for stopping the Snort IDS on a specific IP

        :param emulation_env_config: the emulation env configuration
        :param ip: the ip of the container
        :param logger: the logger to use for logging
        :return: None
        """
        SnortIDSController.start_snort_manager(emulation_env_config=emulation_env_config, ip=ip, logger=logger)
        ids_monitor_dto = SnortIDSController.get_snort_idses_monitor_threads_statuses_by_ip_and_port(
            port=emulation_env_config.snort_ids_manager_config.snort_ids_manager_port, ip=ip)
        if ids_monitor_dto.snort_ids_running:
            logger.info(f"Snort IDS is running on {ip}, stopping it.")
            # Open a gRPC session
            with grpc.insecure_channel(
                    f'{ip}:'
                    f'{emulation_env_config.snort_ids_manager_config.snort_ids_manager_port}',
                    options=constants.GRPC_SERVERS.GRPC_OPTIONS) as channel:
                stub = csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc.SnortIdsManagerStub(channel)
                csle_collector.snort_ids_manager.query_snort_ids_manager.stop_snort_ids(stub=stub)

    @staticmethod
    def start_snort_managers(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                             logger: logging.Logger) -> None:
        """
        Utility function for starting snort IDS managers

        :param emulation_env_config: the emulation env config
        :param physical_server_ip: the physical server ip
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    SnortIDSController.start_snort_manager(emulation_env_config=emulation_env_config,
                                                           ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def start_snort_manager(emulation_env_config: EmulationEnvConfig, ip: str, logger: logging.Logger) -> None:
        """
        Utility function for starting the snort IDS manager on a specific IP

        :param emulation_env_config: the emulation env config
        :param ip: IP of the container
        :param logger: the logger to use for logging
        :return: None
        """
        # Connect
        EmulationUtil.connect_admin(emulation_env_config=emulation_env_config, ip=ip)

        # Check if ids_manager is already running
        cmd = (constants.COMMANDS.PS_AUX + " | " + constants.COMMANDS.GREP + constants.COMMANDS.SPACE_DELIM +
               constants.TRAFFIC_COMMANDS.SNORT_IDS_MANAGER_FILE_NAME)
        o, e, _ = EmulationUtil.execute_ssh_cmd(cmd=cmd, conn=emulation_env_config.get_connection(ip=ip))

        if constants.COMMANDS.SEARCH_SNORT_IDS_MANAGER not in str(o):
            logger.info(f"Starting Snort IDS manager on node {ip}")

            # Stop old background job if running
            cmd = (constants.COMMANDS.SUDO + constants.COMMANDS.SPACE_DELIM + constants.COMMANDS.PKILL +
                   constants.COMMANDS.SPACE_DELIM + constants.TRAFFIC_COMMANDS.SNORT_IDS_MANAGER_FILE_NAME)
            o, e, _ = EmulationUtil.execute_ssh_cmd(
                cmd=cmd, conn=emulation_env_config.get_connection(ip=ip))

            # Start the ids_manager
            cmd = constants.COMMANDS.START_SNORT_IDS_MANAGER.format(
                emulation_env_config.snort_ids_manager_config.snort_ids_manager_port,
                emulation_env_config.snort_ids_manager_config.snort_ids_manager_log_dir,
                emulation_env_config.snort_ids_manager_config.snort_ids_manager_log_file,
                emulation_env_config.snort_ids_manager_config.snort_ids_manager_max_workers)
            o, e, _ = EmulationUtil.execute_ssh_cmd(
                cmd=cmd, conn=emulation_env_config.get_connection(ip=ip))
            time.sleep(2)

    @staticmethod
    def stop_snort_managers(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                            logger: logging.Logger) -> None:
        """
        Utility function for stopping snort IDS managers

        :param emulation_env_config: the emulation env config
        :param physical_server_ip: the physical server ip
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    SnortIDSController.stop_snort_manager(emulation_env_config=emulation_env_config,
                                                          ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def stop_snort_manager(emulation_env_config: EmulationEnvConfig, ip: str, logger: logging.Logger) -> None:
        """
        Utility function for stopping the snort IDS manager on a specific IP

        :param emulation_env_config: the emulation env config
        :param ip: the IP of the container
        :param logger: the logger to use for logging
        :return: None
        """
        # Connect
        EmulationUtil.connect_admin(emulation_env_config=emulation_env_config, ip=ip)
        logger.info(f"Stopping Snort IDS manager on node {ip}")

        cmd = (constants.COMMANDS.SUDO + constants.COMMANDS.SPACE_DELIM + constants.COMMANDS.PKILL +
               constants.COMMANDS.SPACE_DELIM + constants.TRAFFIC_COMMANDS.SNORT_IDS_MANAGER_FILE_NAME)
        o, e, _ = EmulationUtil.execute_ssh_cmd(cmd=cmd, conn=emulation_env_config.get_connection(ip=ip))

        time.sleep(2)

    @staticmethod
    def start_snort_idses_monitor_threads(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                                          logger: logging.Logger) -> None:
        """
        A method that sends a request to the SnortIDSManager on every container that runs
        an IDS to start the IDS manager and the monitor thread

        :param emulation_env_config: the emulation env config
        :param physical_server_ip: the ip of the physical server
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    logger.info(f"Starting Snort IDS monitor thread on IP: {c.docker_gw_bridge_ip}")
                    SnortIDSController.start_snort_idses_monitor_thread(emulation_env_config=emulation_env_config,
                                                                        ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def start_snort_idses_monitor_thread(emulation_env_config: EmulationEnvConfig, ip: str,
                                         logger: logging.Logger) -> None:
        """
        A method that sends a request to the SnortIDSManager on a specific container that runs
        an IDS to start the IDS manager and the monitor thread

        :param emulation_env_config: the emulation env config
        :param ip: the ip of the container
        :param logger: the logger to use for logging
        :return: None
        """
        SnortIDSController.start_snort_manager(emulation_env_config=emulation_env_config, ip=ip, logger=logger)

        ids_monitor_dto = SnortIDSController.get_snort_idses_monitor_threads_statuses_by_ip_and_port(
            port=emulation_env_config.snort_ids_manager_config.snort_ids_manager_port, ip=ip)
        if not ids_monitor_dto.monitor_running:
            logger.info(f"Snort IDS monitor thread is not running on {ip}, starting it.")
            # Open a gRPC session
            with grpc.insecure_channel(
                    f'{ip}:'
                    f'{emulation_env_config.snort_ids_manager_config.snort_ids_manager_port}',
                    options=constants.GRPC_SERVERS.GRPC_OPTIONS) as channel:
                stub = csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc.SnortIdsManagerStub(channel)
                csle_collector.snort_ids_manager.query_snort_ids_manager.start_snort_ids_monitor(
                    stub=stub, kafka_ip=emulation_env_config.kafka_config.container.get_ips()[0],
                    kafka_port=emulation_env_config.kafka_config.kafka_port,
                    log_file_path=csle_collector_constants.SNORT_IDS_ROUTER.SNORT_FAST_LOG_FILE,
                    time_step_len_seconds=emulation_env_config.snort_ids_manager_config.time_step_len_seconds)

    @staticmethod
    def stop_snort_idses_monitor_threads(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                                         logger: logging.Logger) -> None:
        """
        A method that sends a request to the SnortIDSManager on every container that runs
        an IDS to stop the monitor threads

        :param emulation_env_config: the emulation env config
        :param physical_server_ip: the physical server ip
        :param logger: the logger to use for logging
        :return: None
        """
        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    SnortIDSController.stop_snort_idses_monitor_thread(emulation_env_config=emulation_env_config,
                                                                       ip=c.docker_gw_bridge_ip, logger=logger)

    @staticmethod
    def stop_snort_idses_monitor_thread(emulation_env_config: EmulationEnvConfig, ip: str,
                                        logger: logging.Logger) -> None:
        """
        A method that sends a request to the SnortIDSManager on a specific container that runs
        an IDS to stop the monitor threads

        :param emulation_env_config: the emulation env config
        :param ip: the IP of the container
        :param logger: the logger to use for logging
        :return: None
        """
        SnortIDSController.start_snort_manager(emulation_env_config=emulation_env_config, ip=ip, logger=logger)
        # Open a gRPC session
        with grpc.insecure_channel(
                f'{ip}:'
                f'{emulation_env_config.snort_ids_manager_config.snort_ids_manager_port}',
                options=constants.GRPC_SERVERS.GRPC_OPTIONS) as channel:
            stub = csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc.SnortIdsManagerStub(channel)
            logger.info(f"Stopping the Snort IDS monitor thread on {ip}.")
            csle_collector.snort_ids_manager.query_snort_ids_manager.stop_snort_ids_monitor(stub=stub)

    @staticmethod
    def get_snort_idses_monitor_threads_statuses(emulation_env_config: EmulationEnvConfig, physical_server_ip: str,
                                                 logger: logging.Logger, start_if_stopped: bool = True) -> \
            List[csle_collector.snort_ids_manager.snort_ids_manager_pb2.SnortIdsMonitorDTO]:
        """
        A method that sends a request to the SnortIDSManager on every container to get the status of the
        IDS monitor thread

        :param emulation_env_config: the emulation config
        :param start_if_stopped: whether to start the IDS monitor if it is stopped
        :param physical_server_ip: the physical server IP
        :param logger: the logger to use for logging
        :return: List of monitor thread statuses
        """
        statuses = []
        if start_if_stopped:
            SnortIDSController.start_snort_managers(emulation_env_config=emulation_env_config,
                                                    physical_server_ip=physical_server_ip,
                                                    logger=logger)

        for c in emulation_env_config.containers_config.containers:
            if c.physical_host_ip != physical_server_ip:
                continue
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    status = SnortIDSController.get_snort_idses_monitor_threads_statuses_by_ip_and_port(
                        port=emulation_env_config.snort_ids_manager_config.snort_ids_manager_port,
                        ip=c.docker_gw_bridge_ip)
                    statuses.append(status)
        return statuses

    @staticmethod
    def get_snort_idses_monitor_threads_statuses_by_ip_and_port(port: int, ip: str) \
            -> csle_collector.snort_ids_manager.snort_ids_manager_pb2.SnortIdsMonitorDTO:
        """
        A method that sends a request to the SnortIDSManager with a specific port and ip
        to get the status of the IDS monitor thread

        :param port: the port of the SnortIDSManager
        :param ip: the ip of the SnortIDSManager
        :return: the status of the SnortIDSManager
        """
        with grpc.insecure_channel(f'{ip}:{port}', options=constants.GRPC_SERVERS.GRPC_OPTIONS) as channel:
            stub = csle_collector.snort_ids_manager.snort_ids_manager_pb2_grpc.SnortIdsManagerStub(channel)
            status = \
                csle_collector.snort_ids_manager.query_snort_ids_manager.get_snort_ids_monitor_status(
                    stub=stub)
            return status

    @staticmethod
    def get_snort_ids_managers_ips(emulation_env_config: EmulationEnvConfig) -> List[str]:
        """
        A method that extracts the IPs of the snort IDS managers in a given emulation

        :param emulation_env_config: the emulation env config
        :return: the list of IP addresses
        """
        ips = []
        for c in emulation_env_config.containers_config.containers:
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    ips.append(c.docker_gw_bridge_ip)
        return ips

    @staticmethod
    def get_snort_idses_managers_ports(emulation_env_config: EmulationEnvConfig) -> List[int]:
        """
        A method that extracts the ports of the snort IDS managers in a given emulation

        :param emulation_env_config: the emulation env config
        :return: the list of ports
        """
        ports = []
        for c in emulation_env_config.containers_config.containers:
            for ids_image in constants.CONTAINER_IMAGES.SNORT_IDS_IMAGES:
                if ids_image in c.name:
                    ports.append(emulation_env_config.snort_ids_manager_config.snort_ids_manager_port)
        return ports

    @staticmethod
    def get_snort_managers_info(emulation_env_config: EmulationEnvConfig, active_ips: List[str],
                                logger: logging.Logger, physical_server_ip: str) -> SnortIdsManagersInfo:
        """
        Extracts the information of the Snort managers for a given emulation

        :param emulation_env_config: the configuration of the emulation
        :param active_ips: list of active IPs
        :param physical_server_ip: the IP of the physical server
        :param logger: the logger to use for logging
        :return: a DTO with the status of the Snort managers
        """
        snort_ids_managers_ips = SnortIDSController.get_snort_ids_managers_ips(
            emulation_env_config=emulation_env_config)
        snort_ids_managers_ports = \
            SnortIDSController.get_snort_idses_managers_ports(emulation_env_config=emulation_env_config)
        snort_managers_statuses = []
        snort_managers_running = []
        for ip in snort_ids_managers_ips:
            if ip not in active_ips or not EmulationUtil.physical_ip_match(
                    emulation_env_config=emulation_env_config, ip=ip, physical_host_ip=physical_server_ip):
                continue
            running = False
            status = None
            try:
                status = SnortIDSController.get_snort_idses_monitor_threads_statuses_by_ip_and_port(
                    port=emulation_env_config.snort_ids_manager_config.snort_ids_manager_port, ip=ip)
                running = True
            except Exception as e:
                logger.debug(
                    f"Could not fetch Snort IDS manager status on IP:{ip}, error: {str(e)}, {repr(e)}")
            if status is not None:
                snort_managers_statuses.append(status)
            else:
                util = csle_collector.snort_ids_manager.snort_ids_manager_util.SnortIdsManagerUtil
                snort_managers_statuses.append(util.snort_ids_monitor_dto_empty())
            snort_managers_running.append(running)
        execution_id = emulation_env_config.execution_id
        emulation_name = emulation_env_config.name
        snort_manager_info_dto = SnortIdsManagersInfo(
            snort_ids_managers_running=snort_managers_running, ips=snort_ids_managers_ips,
            ports=snort_ids_managers_ports, execution_id=execution_id, emulation_name=emulation_name,
            snort_ids_managers_statuses=snort_managers_statuses)
        return snort_manager_info_dto
