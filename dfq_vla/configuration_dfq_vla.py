from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers import DINOv3ViTConfig, Qwen3Config


VISION_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DFQVLAConfig(PretrainedConfig):
    model_type = "dfq_vla"
    sub_configs = {"vision_config": DINOv3ViTConfig, "text_config": Qwen3Config}
    
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            projector_hidden_act="gelu",
            ignore_index=-100,
            # carried over from qwen3_vl
            image_token_id=151655,
            # video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            tie_word_embeddings=True,
            # Trajectory encoding config
            traj_vocab_size: int = 768,
            tokens_per_history_traj: int = 16,  # 16 waypoints, 1 embedding each
            traj_token_start_idx: int = None,  # Set during tokenizer init
            traj_token_ids: dict = None,  # Mapping for special tokens
            traj_input_dim: int = 4,  # Per-waypoint feature dim (xyz + yaw)
            # Action head config (for future trajectory prediction)
            num_action_queries: int = 64,  # Number of future waypoints
            num_action_layers: int = 4,  # Transformer decoder layers
            action_nhead: int = 16,  # Attention heads
            action_dim_feedforward: int = 4096,  # FFN dimension
            action_dropout: float = 0.1,  # Dropout
            # Flex Scene Encoder config
            use_flex_scene_encoder: bool = True,  # Enable Flex encoder
            num_cameras: int = 4,  # Number of camera views
            num_timestamps: int = 4,  # Number of timestamps
            num_scene_tokens: int = 800,  # K = 50 per image × 16 images
            flex_encoder_layers: int = 4,  # Transformer layers (balanced for ~1B model)
            flex_encoder_heads: int = 8,  # Attention heads
            flex_encoder_dim_feedforward: int = 3072,  # FFN dimension
            flex_encoder_dropout: float = 0.1,  # Dropout
            **kwargs
    ):

        super().__init__(**kwargs)

        # Initialize sub-configs
        if vision_config is None:
            # Defaults to DINOv3 ViT-B/16 config if not provided
            self.vision_config = AutoConfig.from_pretrained(VISION_MODEL_ID)
        elif isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        else:
            self.vision_config = vision_config

        if text_config is None:
            self.text_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        elif isinstance(text_config, dict):
            self.text_config = Qwen3Config(**text_config)
        else:
            self.text_config = text_config

        self.hidden_size = self.text_config.hidden_size
        self.vision_hidden_size = self.vision_config.hidden_size  # e.g. 768 for DINOv3 ViT-B/16
        self.projector_hidden_act = projector_hidden_act
        self.ignore_index = ignore_index
        self.vocab_size = self.text_config.vocab_size
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        
        # Trajectory encoding config
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.traj_token_start_idx = traj_token_start_idx
        self.traj_token_ids = traj_token_ids
        self.traj_input_dim = traj_input_dim
        
        # Action head config
        self.num_action_queries = num_action_queries
        self.num_action_layers = num_action_layers
        self.action_nhead = action_nhead
        self.action_dim_feedforward = action_dim_feedforward
        self.action_dropout = action_dropout
        
        # Flex Scene Encoder config
        self.use_flex_scene_encoder = use_flex_scene_encoder
        self.num_cameras = num_cameras
        self.num_timestamps = num_timestamps
        self.num_scene_tokens = num_scene_tokens
        self.flex_encoder_layers = flex_encoder_layers
        self.flex_encoder_heads = flex_encoder_heads
        self.flex_encoder_dim_feedforward = flex_encoder_dim_feedforward
        self.flex_encoder_dropout = flex_encoder_dropout
        
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

