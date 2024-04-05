import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    Encoder for Numpy arrays to JSON
    """
    def default(self, obj):
        """
        The default encoding of numpy datatypes for JSON

        :param obj: the object to encoded
        :return: the encoded object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
