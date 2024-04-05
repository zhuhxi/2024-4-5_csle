# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import csle_collector.elk_manager.elk_manager_pb2 as elk__manager__pb2


class ElkManagerStub(object):
    """Interface exported by the server
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getElkStatus = channel.unary_unary(
                '/ElkManager/getElkStatus',
                request_serializer=elk__manager__pb2.GetElkStatusMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.stopElk = channel.unary_unary(
                '/ElkManager/stopElk',
                request_serializer=elk__manager__pb2.StopElkMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.startElk = channel.unary_unary(
                '/ElkManager/startElk',
                request_serializer=elk__manager__pb2.StartElkMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.stopElastic = channel.unary_unary(
                '/ElkManager/stopElastic',
                request_serializer=elk__manager__pb2.StopElasticMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.startElastic = channel.unary_unary(
                '/ElkManager/startElastic',
                request_serializer=elk__manager__pb2.StartElasticMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.stopLogstash = channel.unary_unary(
                '/ElkManager/stopLogstash',
                request_serializer=elk__manager__pb2.StopLogstashMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.startLogstash = channel.unary_unary(
                '/ElkManager/startLogstash',
                request_serializer=elk__manager__pb2.StartLogstashMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.stopKibana = channel.unary_unary(
                '/ElkManager/stopKibana',
                request_serializer=elk__manager__pb2.StopKibanaMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )
        self.startKibana = channel.unary_unary(
                '/ElkManager/startKibana',
                request_serializer=elk__manager__pb2.StartKibanaMsg.SerializeToString,
                response_deserializer=elk__manager__pb2.ElkDTO.FromString,
                )


class ElkManagerServicer(object):
    """Interface exported by the server
    """

    def getElkStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def stopElk(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def startElk(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def stopElastic(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def startElastic(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def stopLogstash(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def startLogstash(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def stopKibana(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def startKibana(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ElkManagerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getElkStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.getElkStatus,
                    request_deserializer=elk__manager__pb2.GetElkStatusMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'stopElk': grpc.unary_unary_rpc_method_handler(
                    servicer.stopElk,
                    request_deserializer=elk__manager__pb2.StopElkMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'startElk': grpc.unary_unary_rpc_method_handler(
                    servicer.startElk,
                    request_deserializer=elk__manager__pb2.StartElkMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'stopElastic': grpc.unary_unary_rpc_method_handler(
                    servicer.stopElastic,
                    request_deserializer=elk__manager__pb2.StopElasticMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'startElastic': grpc.unary_unary_rpc_method_handler(
                    servicer.startElastic,
                    request_deserializer=elk__manager__pb2.StartElasticMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'stopLogstash': grpc.unary_unary_rpc_method_handler(
                    servicer.stopLogstash,
                    request_deserializer=elk__manager__pb2.StopLogstashMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'startLogstash': grpc.unary_unary_rpc_method_handler(
                    servicer.startLogstash,
                    request_deserializer=elk__manager__pb2.StartLogstashMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'stopKibana': grpc.unary_unary_rpc_method_handler(
                    servicer.stopKibana,
                    request_deserializer=elk__manager__pb2.StopKibanaMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
            'startKibana': grpc.unary_unary_rpc_method_handler(
                    servicer.startKibana,
                    request_deserializer=elk__manager__pb2.StartKibanaMsg.FromString,
                    response_serializer=elk__manager__pb2.ElkDTO.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ElkManager', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ElkManager(object):
    """Interface exported by the server
    """

    @staticmethod
    def getElkStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/getElkStatus',
            elk__manager__pb2.GetElkStatusMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def stopElk(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/stopElk',
            elk__manager__pb2.StopElkMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def startElk(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/startElk',
            elk__manager__pb2.StartElkMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def stopElastic(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/stopElastic',
            elk__manager__pb2.StopElasticMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def startElastic(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/startElastic',
            elk__manager__pb2.StartElasticMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def stopLogstash(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/stopLogstash',
            elk__manager__pb2.StopLogstashMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def startLogstash(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/startLogstash',
            elk__manager__pb2.StartLogstashMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def stopKibana(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/stopKibana',
            elk__manager__pb2.StopKibanaMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def startKibana(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ElkManager/startKibana',
            elk__manager__pb2.StartKibanaMsg.SerializeToString,
            elk__manager__pb2.ElkDTO.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
