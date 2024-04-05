from typing import List, Dict, Any
from csle_common.dao.simulation_config.observation_space_config import ObservationSpaceConfig
from csle_base.json_serializable import JSONSerializable


class JointObservationSpaceConfig(JSONSerializable):
    """
    DTO representing a joint observation space configuration
    """

    def __init__(self, observation_spaces: List[ObservationSpaceConfig]):
        """
        Initializes the DTO

        :param observation_spaces: list of observation spaces
        """
        self.observation_spaces = observation_spaces

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "JointObservationSpaceConfig":
        """
        Converts a dict representation to an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = JointObservationSpaceConfig(
            observation_spaces=list(map(lambda x: ObservationSpaceConfig.from_dict(x), d["observation_spaces"]))
        )
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dict representation
        
        :return: a dict representation of the object
        """
        d = {}
        d["observation_spaces"] = list(map(lambda x: x.to_dict(), self.observation_spaces))
        return d

    def __str__(self):
        """
        :return: a string representation of the object
        """
        return f"observation spaces: {list(map(lambda x: str(x), self.observation_spaces))}"

    @staticmethod
    def from_json_file(json_file_path: str) -> "JointObservationSpaceConfig":
        """
        Reads a json file and converts it to a DTO

        :param json_file_path: the json file path
        :return: the converted DTO
        """
        import io
        import json
        with io.open(json_file_path, 'r') as f:
            json_str = f.read()
        return JointObservationSpaceConfig.from_dict(json.loads(json_str))
