import torch
import torch.nn.functional
from attrdict import AttrDict


def _calculate_fan(dimensions, conv_params, axis):
    stride = conv_params.get('stride', [1, 1])[axis]
    padding = conv_params.get('padding', [0, 0])[axis]
    kernel_size = conv_params.get('kernel_size')[axis]
    dillation = conv_params.get('dillation', [1, 1])[axis]

    return torch.floor(
        (dimensions + 2 * padding - dillation * (kernel_size - 1) - 1) / stride + 1
    ).to(dtype=torch.long)


class Model(torch.nn.Module):
    def __init__(self, num_mel_bins, hidden_size, num_layers, num_tokens):
        super(Model, self).__init__()

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.num_mel_bins = num_mel_bins

        conv1_params = AttrDict(
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": [21, 11],
                "stride": [1, 1]
            })
        conv2_params = AttrDict(
            {
                "in_channels": conv1_params['out_channels'],
                "out_channels": 64,
                "kernel_size": [11, 11],
                "stride": [1, 3]
            }
        )

        self.conv1_params = conv1_params
        self.conv2_params = conv2_params

        # [batch_size x 1 X num_mel_bins x time] -> [batch_size x conv2.out_channels x fan_num_mel_bins x fan_time]
        self.conv = torch.nn.Sequential(
            # write your code here

            # CONV 1
            torch.nn.Conv2d(**self.conv1_params, bias=False),
            # BATCH NORM 1
            torch.nn.BatchNorm2d(num_features=self.conv1_params['out_channels'], momentum=0.9),
            # RELU
            torch.nn.ReLU(inplace=True),

            # CONV 2
            torch.nn.Conv2d(**self.conv2_params, bias=False),
            # BATCH NORM 2
            torch.nn.BatchNorm2d(num_features=self.conv2_params['out_channels'], momentum=0.9),
            # RELU
            torch.nn.ReLU(inplace=True),
        )

        # YOUR CODE
        fan_num_mel_bins = _calculate_fan(
            _calculate_fan(torch.tensor(num_mel_bins), conv1_params, axis=0),
            conv2_params, axis=0
        ).item()
        rnn_input_size = self.conv2_params['out_channels'] * fan_num_mel_bins

        # 4 layers bidirectional lstm
        # YOUR CODE
        # [batch_size x fan_time x rnn_input_size] -> [batch_size x fan_time x num_directions * hidden_size], (h_n, c_n)
        self.lstm = torch.nn.LSTM(
            input_size=rnn_input_size, hidden_size=hidden_size, num_layers=self.num_layers,
            bias=True, batch_first=True, bidirectional=self.bidirectional
        )

        # YOUR CODE
        # [batch_size x num_directions * hidden_size] -> [batch_size, num_tokens]
        self.output_layer = torch.nn.Linear(self.num_directions * self.hidden_size, self.num_tokens)

    def forward(self, inputs, seq_lens, state=None):
        """
            Input shape:
                audio: 3D tensor with shape (batch_size, num_mel_bins, num_timesteps)
                sequence_lengths: 1D tensor with shape (batch_size)
            Returns:
                3D tensor with shape (new_num_timesteps, batch_size, alphabet_len)
                1D tensor with shape (batch_size)
            """

        outputs = inputs.unsqueeze(1)  # conv2d input should be four-dimensional

        ### write your code here ###
        seq_lens = _calculate_fan(_calculate_fan(seq_lens, self.conv1_params, axis=1), self.conv2_params, axis=1)

        outputs = self.conv(outputs)

        outputs = self.transpose_and_reshape(outputs)
        outputs, (h_n, c_n) = self.lstm(outputs)

        outputs = self.output_layer(outputs)
        outputs = torch.transpose(outputs, 0, 1)
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)

        return outputs, seq_lens

    @staticmethod
    def transpose_and_reshape(inputs):
        """ This function will be very useful for converting the output of a convolutional layer
            to the input of a lstm layer

            Input shape:
                inputs: 4D tensor with shape (batch_size, num_filters, num_features, num_timesteps)
            Returns:
                3D tensor with shape (batch_size, num_timesteps, new_num_features)
            """
        # reshape # YOUR CODE
        # (batch_size, num_filters * num_features, num_timesteps)
        outputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3])
        # transpose # YOUR CODE
        # (batch_size, num_timesteps, new_num_features)
        outputs = torch.transpose(outputs, 1, 2)
        return outputs

    @staticmethod
    def get_new_seq_lens(seq_lens, conv1_kernel_size, conv1_stride, conv2_kernel_size, conv2_stride):
        """ Compute sequence_lengths after convolutions
            """
        # write your code here
        seq_lens = _calculate_fan(seq_lens, {'kernel_size': [conv1_kernel_size], 'stride': [conv1_stride]}, axis=0)
        seq_lens = _calculate_fan(seq_lens, {'kernel_size': [conv2_kernel_size], 'stride': [conv2_stride]}, axis=0)
        return seq_lens
