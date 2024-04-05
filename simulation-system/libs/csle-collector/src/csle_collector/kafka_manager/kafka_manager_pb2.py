# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: kafka_manager.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13kafka_manager.proto\"\x0e\n\x0cStopKafkaMsg\"\x0f\n\rStartKafkaMsg\"\x13\n\x11GetKafkaStatusMsg\"+\n\x08KafkaDTO\x12\x0f\n\x07running\x18\x01 \x01(\x08\x12\x0e\n\x06topics\x18\x02 \x03(\t\"b\n\x0e\x43reateTopicMsg\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\npartitions\x18\x02 \x01(\x05\x12\x10\n\x08replicas\x18\x03 \x01(\x05\x12\x1c\n\x14retention_time_hours\x18\x04 \x01(\x05\"\x1e\n\x0e\x44\x65leteTopicMsg\x12\x0c\n\x04name\x18\x01 \x01(\t2\xef\x01\n\x0cKafkaManager\x12\x31\n\x0egetKafkaStatus\x12\x12.GetKafkaStatusMsg\x1a\t.KafkaDTO\"\x00\x12\'\n\tstopKafka\x12\r.StopKafkaMsg\x1a\t.KafkaDTO\"\x00\x12)\n\nstartKafka\x12\x0e.StartKafkaMsg\x1a\t.KafkaDTO\"\x00\x12+\n\x0b\x63reateTopic\x12\x0f.CreateTopicMsg\x1a\t.KafkaDTO\"\x00\x12+\n\x0b\x64\x65leteTopic\x12\x0f.DeleteTopicMsg\x1a\t.KafkaDTO\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'kafka_manager_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _STOPKAFKAMSG._serialized_start=23
  _STOPKAFKAMSG._serialized_end=37
  _STARTKAFKAMSG._serialized_start=39
  _STARTKAFKAMSG._serialized_end=54
  _GETKAFKASTATUSMSG._serialized_start=56
  _GETKAFKASTATUSMSG._serialized_end=75
  _KAFKADTO._serialized_start=77
  _KAFKADTO._serialized_end=120
  _CREATETOPICMSG._serialized_start=122
  _CREATETOPICMSG._serialized_end=220
  _DELETETOPICMSG._serialized_start=222
  _DELETETOPICMSG._serialized_end=252
  _KAFKAMANAGER._serialized_start=255
  _KAFKAMANAGER._serialized_end=494
# @@protoc_insertion_point(module_scope)
