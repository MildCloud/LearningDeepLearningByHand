from turtle import forward
from torch import nn


class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

    
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture"""
    def __init__(self, encoder, decoder, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


# def test_kwargs(*args, **kwargs):
#    print(type(kwargs))
#    for v in args:
#       print ('Optional argument (args): ', v)
#    for k, v in kwargs.items():
#       print ('Optional argument %s (kwargs): %s' % (k, v))

# test_kwargs(1, 2, 3, 4, k1=5, k2=6)
# Optional argument (args):  1
# Optional argument (args):  2
# Optional argument (args):  3
# Optional argument (args):  4
# Optional argument k1 (kwargs): 5
# Optional argument k2 (kwargs): 6
