from .ic_controller import ICMAC
from .maddpg_controller import MADDPGMAC
from .non_shared_controller import NonSharedMAC
from .basic_controller import BasicMAC
REGISTRY = {}


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ic_mac"] = ICMAC
