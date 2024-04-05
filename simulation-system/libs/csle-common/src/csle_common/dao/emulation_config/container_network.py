from typing import Dict, Any
from csle_common.util.general_util import GeneralUtil
from csle_base.json_serializable import JSONSerializable


class ContainerNetwork(JSONSerializable):
    """
    DTO representing an IP network of virtual containers
    """

    def __init__(self, name: str, subnet_mask: str, bitmask: str, subnet_prefix: str, interface: str = "eth0"):
        """
        Initializes the DTO

        :param name: the name of the network
        :param subnet_mask: the subnet mask of the network
        :param bitmask: the bitmask of the network
        :param subnet_prefix: the subnet prefix of the network
        :param interface: (optional) the interface of the network
        """
        self.name = name
        self.subnet_mask = subnet_mask
        self.bitmask = bitmask
        self.subnet_prefix = subnet_prefix
        self.interface = interface

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ContainerNetwork":
        """
        Converts a dict representation to an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = ContainerNetwork(
            name=d["name"], subnet_mask=d["subnet_mask"], subnet_prefix=d["subnet_prefix"],
            interface=d["interface"], bitmask=d["bitmask"])
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dict representation
        
        :return: a dict representation of the object
        """
        d = {}
        d["name"] = self.name
        d["subnet_mask"] = self.subnet_mask
        d["subnet_prefix"] = self.subnet_prefix
        d["interface"] = self.interface
        d["bitmask"] = self.bitmask
        return d

    def __str__(self):
        """
        :return: a string representation of the object
        """
        return f"name:{self.name}, subnet_mask:{self.subnet_mask}, subnet_prefix: {self.subnet_prefix}, " \
               f"interface: {self.interface}, bitmask: {self.bitmask}"

    @staticmethod
    def from_json_file(json_file_path: str) -> "ContainerNetwork":
        """
        Reads a json file and converts it to a DTO

        :param json_file_path: the json file path
        :return: the converted DTO
        """
        import io
        import json
        with io.open(json_file_path, 'r') as f:
            json_str = f.read()
        return ContainerNetwork.from_dict(json.loads(json_str))

    def copy(self) -> "ContainerNetwork":
        """
        :return: a copy of the DTO
        """
        return ContainerNetwork.from_dict(self.to_dict())

    def create_execution_config(self, ip_first_octet: int) -> "ContainerNetwork":
        """
        Creates a new config for an execution

        :param ip_first_octet: the first octet of the IP of the new execution
        :return: the new config
        """
        config = self.copy()
        config.name = config.name + f"_{ip_first_octet}"
        config.subnet_mask = GeneralUtil.replace_first_octet_of_ip(ip=config.subnet_mask,
                                                                   ip_first_octet=ip_first_octet)
        config.subnet_prefix = GeneralUtil.replace_first_octet_of_ip(ip=config.subnet_mask,
                                                                     ip_first_octet=ip_first_octet)
        return config

    @staticmethod
    def schema() -> "ContainerNetwork":
        """
        :return: get the schema of the DTO
        """
        return ContainerNetwork(name="", subnet_mask="", bitmask="", subnet_prefix="", interface="")
