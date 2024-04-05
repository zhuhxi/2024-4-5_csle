from typing import List
import logging
import os
import subprocess
from concurrent import futures
import grpc
import socket
import netifaces
import io
import csle_collector.traffic_manager.traffic_manager_pb2_grpc
import csle_collector.traffic_manager.traffic_manager_pb2
import csle_collector.constants.constants as constants


class TrafficManagerServicer(csle_collector.traffic_manager.traffic_manager_pb2_grpc.TrafficManagerServicer):
    """
    gRPC server for managing a traffic generator. Allows to start/stop the script and also to query the
    state of the script.
    """

    def __init__(self) -> None:
        """
        Initializes the server
        """
        logging.basicConfig(filename=f"{constants.LOG_FILES.TRAFFIC_MANAGER_LOG_DIR}"
                                     f"{constants.LOG_FILES.TRAFFIC_MANAGER_LOG_FILE}", level=logging.INFO)
        self.hostname = socket.gethostname()
        try:
            self.ip = netifaces.ifaddresses(constants.INTERFACES.ETH0)[netifaces.AF_INET][0][constants.INTERFACES.ADDR]
        except Exception:
            self.ip = socket.gethostbyname(self.hostname)
        logging.info(f"Setting up TrafficManager hostname: {self.hostname} ip: {self.ip}")

    @staticmethod
    def _get_traffic_status() -> bool:
        """
        Utility method to get the status of the traffic generator script

        :return: status of the traffic generator
        """
        cmd = constants.TRAFFIC_GENERATOR.CHECK_IF_TRAFFIC_GENERATOR_IS_RUNNING
        output = subprocess.run(cmd.split(" "), capture_output=True, text=True).stdout
        running = constants.TRAFFIC_GENERATOR.TRAFFIC_GENERATOR_FILE_NAME in output
        return running

    @staticmethod
    def _read_traffic_script() -> str:
        """
        :return: the traffic generator script file
        """
        try:
            with io.open(f"/{constants.TRAFFIC_GENERATOR.TRAFFIC_GENERATOR_FILE_NAME}", 'r', encoding='utf-8') as f:
                script_file_str = f.read()
                return script_file_str
        except Exception as e:
            logging.info(f"Could not read the script file: {str(e)}, {repr(e)}")
            return ""

    @staticmethod
    def _create_traffic_script(commands: List[str], sleep_time: int) -> None:
        """
        Utility method to create the traffic script based on the list of commands

        :param commands: the list of commands
        :param sleep_time: the sleep time
        :return: None
        """
        # File contents
        script_file = ""
        script_file = script_file + "#!/bin/bash\n"
        script_file = script_file + "while [ 1 ]\n"
        script_file = script_file + "do\n"
        script_file = script_file + "    sleep {}\n".format(sleep_time)
        for cmd in commands:
            script_file = script_file + "    " + cmd + "\n"
            script_file = script_file + "    sleep {}\n".format(sleep_time)
        script_file = script_file + "done\n"

        # Remove old file if exists
        cmd = constants.TRAFFIC_GENERATOR.REMOVE_OLD_TRAFFIC_GENERATOR_FILE
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        logging.info(f"Removed old file, stdout: {result.stdout}, stderr: {result.stderr}")

        # Create file
        cmd = constants.TRAFFIC_GENERATOR.CREATE_TRAFFIC_GENERATOR_FILE
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        logging.info(f"Created new file, stdout: {result.stdout}, stderr: {result.stderr}")

        # Make executable
        cmd = constants.TRAFFIC_GENERATOR.MAKE_TRAFFIC_GENERATOR_FILE_EXECUTABLE
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        logging.info(f"Changed permissions, stdout: {result.stdout}, stderr: {result.stderr}")

        # Write traffic generation script file
        with io.open(f"/{constants.TRAFFIC_GENERATOR.TRAFFIC_GENERATOR_FILE_NAME}", 'w', encoding='utf-8') as f:
            f.write(script_file)

    def getTrafficStatus(self, request: csle_collector.traffic_manager.traffic_manager_pb2.GetTrafficStatusMsg,
                         context: grpc.ServicerContext) \
            -> csle_collector.traffic_manager.traffic_manager_pb2.TrafficDTO:
        """
        Gets the state of the traffic manager

        :param request: the gRPC request
        :param context: the gRPC context
        :return: a TrafficDTO with the state of the traffic manager
        """
        logging.info("Getting traffic manager status")
        running = TrafficManagerServicer._get_traffic_status()
        script_file_str = TrafficManagerServicer._read_traffic_script()
        logging.info("Returning traffic manager status")
        traffic_dto = csle_collector.traffic_manager.traffic_manager_pb2.TrafficDTO(running=running,
                                                                                    script=script_file_str)
        return traffic_dto

    def stopTraffic(self, request: csle_collector.traffic_manager.traffic_manager_pb2.StopTrafficMsg,
                    context: grpc.ServicerContext):
        """
        Stops the traffic generator

        :param request: the gRPC request
        :param context: the gRPC context
        :return: a traffic DTO with the state of the traffic generator
        """
        logging.info("Stopping the traffic generator script")
        cmd = "sudo pkill -f traffic_generator.sh"
        os.system(cmd)
        script_file_str = TrafficManagerServicer._read_traffic_script()
        logging.info("Traffic generator stopped")
        return csle_collector.traffic_manager.traffic_manager_pb2.TrafficDTO(running=False, script=script_file_str)

    def startTraffic(self, request: csle_collector.traffic_manager.traffic_manager_pb2.StartTrafficMsg,
                     context: grpc.ServicerContext) -> csle_collector.traffic_manager.traffic_manager_pb2.TrafficDTO:
        """
        Starts the traffic generator

        :param request: the gRPC request
        :param context: the gRPC context
        :return: a TrafficDTO with the state of the traffic generator
        """
        logging.info(f"Starting the traffic generator, \n sleep_time: {request.sleepTime}, "
                     f"commands:{request.commands}")
        commands = list(request.commands)
        sleep_time = request.sleepTime
        TrafficManagerServicer._create_traffic_script(commands=commands, sleep_time=sleep_time)
        cmd = constants.TRAFFIC_GENERATOR.START_TRAFFIC_GENERATOR_CMD
        os.system(cmd)
        script_file_str = TrafficManagerServicer._read_traffic_script()
        traffic_dto = csle_collector.traffic_manager.traffic_manager_pb2.TrafficDTO(
            running=True, script=script_file_str)
        logging.info("Traffic generator started")
        return traffic_dto


def serve(port: int = 50043, log_dir: str = "/", max_workers: int = 10,
          log_file_name: str = "traffic_manager.log") -> None:
    """
    Starts the gRPC server for managing traffic scripts

    :param port: the port that the server will listen to
    :param log_dir: the directory to write the log file
    :param log_file_name: the file name of the log
    :param max_workers: the maximum number of GRPC workers
    :return: None
    """
    constants.LOG_FILES.TRAFFIC_MANAGER_LOG_DIR = log_dir
    constants.LOG_FILES.TRAFFIC_MANAGER_LOG_FILE = log_file_name
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    csle_collector.traffic_manager.traffic_manager_pb2_grpc.add_TrafficManagerServicer_to_server(
        TrafficManagerServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f"TrafficManager Server Started, Listening on port: {port}")
    server.wait_for_termination()


# Program entrypoint
if __name__ == '__main__':
    serve()
