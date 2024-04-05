"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class StartLogstashMsg(google.protobuf.message.Message):
    """Message that the client sends to stop logstash"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StartLogstashMsg = StartLogstashMsg

@typing_extensions.final
class StartKibanaMsg(google.protobuf.message.Message):
    """Message that the client sends to stop kibana"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StartKibanaMsg = StartKibanaMsg

@typing_extensions.final
class StartElasticMsg(google.protobuf.message.Message):
    """Message that the client sends to stop elasticsearch"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StartElasticMsg = StartElasticMsg

@typing_extensions.final
class StopLogstashMsg(google.protobuf.message.Message):
    """Message that the client sends to stop logstash"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StopLogstashMsg = StopLogstashMsg

@typing_extensions.final
class StopKibanaMsg(google.protobuf.message.Message):
    """Message that the client sends to stop kibana"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StopKibanaMsg = StopKibanaMsg

@typing_extensions.final
class StopElasticMsg(google.protobuf.message.Message):
    """Message that the client sends to stop elasticsearch"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StopElasticMsg = StopElasticMsg

@typing_extensions.final
class StopElkMsg(google.protobuf.message.Message):
    """Message that the client sends to stop the ELK stack"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StopElkMsg = StopElkMsg

@typing_extensions.final
class StartElkMsg(google.protobuf.message.Message):
    """Message that the client sends to start the ELK stack"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___StartElkMsg = StartElkMsg

@typing_extensions.final
class GetElkStatusMsg(google.protobuf.message.Message):
    """Message that the client sends to extract the elk status"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___GetElkStatusMsg = GetElkStatusMsg

@typing_extensions.final
class ElkDTO(google.protobuf.message.Message):
    """Message that the server returns when requested by the client, contains info about the ELK stack"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ELASTICRUNNING_FIELD_NUMBER: builtins.int
    KIBANARUNNING_FIELD_NUMBER: builtins.int
    LOGSTASHRUNNING_FIELD_NUMBER: builtins.int
    elasticRunning: builtins.bool
    kibanaRunning: builtins.bool
    logstashRunning: builtins.bool
    def __init__(
        self,
        *,
        elasticRunning: builtins.bool = ...,
        kibanaRunning: builtins.bool = ...,
        logstashRunning: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["elasticRunning", b"elasticRunning", "kibanaRunning", b"kibanaRunning", "logstashRunning", b"logstashRunning"]) -> None: ...

global___ElkDTO = ElkDTO
