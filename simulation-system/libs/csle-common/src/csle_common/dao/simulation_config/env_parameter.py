from typing import Dict, Any
from csle_base.json_serializable import JSONSerializable


class EnvParameter(JSONSerializable):
    """
    DTO representing the a general parameter of a simulation environment
    """

    def __init__(self, id: int, name: str, descr: str):
        """
        Initializes the DTO

        :param id: the id of the parameter
        :param name: the name of the parameter
        :param descr: a description of the parameter
        """
        self.id = id
        self.name = name
        self.descr = descr

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EnvParameter":
        """
        Converts a dict representation to an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = EnvParameter(
            id=d["id"], name=d["name"], descr=d["descr"]
        )
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dict representation
        
        :return: a dict representation of the object
        """
        d: Dict[str, Any] = {}
        d["id"] = self.id
        d["name"] = self.name
        d["descr"] = self.descr
        return d

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return f"id:{self.id}, name:{self.name}, descr:{self.descr}"

    @staticmethod
    def from_json_file(json_file_path: str) -> "EnvParameter":
        """
        Reads a json file and converts it to a DTO

        :param json_file_path: the json file path
        :return: the converted DTO
        """
        import io
        import json
        with io.open(json_file_path, 'r') as f:
            json_str = f.read()
        return EnvParameter.from_dict(json.loads(json_str))
