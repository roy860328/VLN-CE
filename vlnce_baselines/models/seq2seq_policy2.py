import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder2 import InstructionEncoder2
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from vlnce_baselines.models.policy import BasePolicy

from pytorch_pretrained_bert import BertModel, BertConfig, BertTokenizer

class Seq2SeqPolicy2(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class Seq2SeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.

    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder2(model_config.INSTRUCTION_ENCODER)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()

        # bert
        # import gzip
        # with gzip.open(model_config.INSTRUCTION_ENCODER.embedding_file, "rt") as f:
        #     import json
        #     embeddings = torch.tensor(json.load(f))
        # if model_config.INSTRUCTION_ENCODER.use_pretrained_embeddings:
        #     self.embedding_layer = nn.Embedding.from_pretrained(
        #         embeddings=embeddings,
        #         freeze=not model_config.INSTRUCTION_ENCODER.fine_tune_embeddings,
        #     )
        # else:  # each embedding initialized to sampled Gaussian
        #     self.embedding_layer = nn.Embedding(
        #         num_embeddings=model_config.INSTRUCTION_ENCODER.vocab_size,
        #         embedding_dim=model_config.INSTRUCTION_ENCODER.embedding_size,
        #         padding_idx=0,
        #     )

        # configuration = BertConfig(hidden_size=model_config.INSTRUCTION_ENCODER.hidden_size,
        #                            vocab_size_or_config_json_file=model_config.INSTRUCTION_ENCODER.vocab_size,
        #                            num_attention_heads=8,
        #                            )
        # self.bert = BertModel(configuration)
        import json
        with open("index_to_word.json") as f:
            self.idx_to_word = list(json.load(f)["word"].keys())
            print(self.idx_to_word)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def _get_bert_embedding(self, observations):
        instruction = observations["instruction"].int()[0]
        # embedded = self.embedding_layer(instruction)
        token = list(map(lambda ind: self.idx_to_word[ind], instruction))
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(token)
        tokens_tensor = torch.tensor([indexed_tokens])
        encoded_layers, _ = self.bert(tokens_tensor)
        return encoded_layers[11]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        ### instruction
        # instruction_embedding = self.instruction_encoder(observations)
        instruction_embedding = self._get_bert_embedding(observations)

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        # print("depth_embedding: ", depth_embedding)
        # print("depth_embedding: ", depth_embedding.size())

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), observations["progress"], reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_hidden_states
