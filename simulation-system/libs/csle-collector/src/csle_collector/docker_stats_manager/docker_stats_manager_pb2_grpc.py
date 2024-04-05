# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import csle_collector.docker_stats_manager.docker_stats_manager_pb2 as docker__stats__manager__pb2


class DockerStatsManagerStub(object):
    """Interface exported by the server
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getDockerStatsMonitorStatus = channel.unary_unary(
                '/DockerStatsManager/getDockerStatsMonitorStatus',
                request_serializer=docker__stats__manager__pb2.GetDockerStatsMonitorStatusMsg.SerializeToString,
                response_deserializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
                )
        self.stopDockerStatsMonitor = channel.unary_unary(
                '/DockerStatsManager/stopDockerStatsMonitor',
                request_serializer=docker__stats__manager__pb2.StopDockerStatsMonitorMsg.SerializeToString,
                response_deserializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
                )
        self.startDockerStatsMonitor = channel.unary_unary(
                '/DockerStatsManager/startDockerStatsMonitor',
                request_serializer=docker__stats__manager__pb2.StartDockerStatsMonitorMsg.SerializeToString,
                response_deserializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
                )


class DockerStatsManagerServicer(object):
    """Interface exported by the server
    """

    def getDockerStatsMonitorStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def stopDockerStatsMonitor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def startDockerStatsMonitor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DockerStatsManagerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getDockerStatsMonitorStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.getDockerStatsMonitorStatus,
                    request_deserializer=docker__stats__manager__pb2.GetDockerStatsMonitorStatusMsg.FromString,
                    response_serializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.SerializeToString,
            ),
            'stopDockerStatsMonitor': grpc.unary_unary_rpc_method_handler(
                    servicer.stopDockerStatsMonitor,
                    request_deserializer=docker__stats__manager__pb2.StopDockerStatsMonitorMsg.FromString,
                    response_serializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.SerializeToString,
            ),
            'startDockerStatsMonitor': grpc.unary_unary_rpc_method_handler(
                    servicer.startDockerStatsMonitor,
                    request_deserializer=docker__stats__manager__pb2.StartDockerStatsMonitorMsg.FromString,
                    response_serializer=docker__stats__manager__pb2.DockerStatsMonitorDTO.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'DockerStatsManager', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DockerStatsManager(object):
    """Interface exported by the server
    """

    @staticmethod
    def getDockerStatsMonitorStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DockerStatsManager/getDockerStatsMonitorStatus',
            docker__stats__manager__pb2.GetDockerStatsMonitorStatusMsg.SerializeToString,
            docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def stopDockerStatsMonitor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DockerStatsManager/stopDockerStatsMonitor',
            docker__stats__manager__pb2.StopDockerStatsMonitorMsg.SerializeToString,
            docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def startDockerStatsMonitor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DockerStatsManager/startDockerStatsMonitor',
            docker__stats__manager__pb2.StartDockerStatsMonitorMsg.SerializeToString,
            docker__stats__manager__pb2.DockerStatsMonitorDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
