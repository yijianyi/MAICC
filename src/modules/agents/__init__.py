from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .transformer_agent import TransformerAgent
from .tabx_transformer_agent import TabxTransformerAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["transformer"] = TransformerAgent
REGISTRY["tabx_transformer"] = TabxTransformerAgent
