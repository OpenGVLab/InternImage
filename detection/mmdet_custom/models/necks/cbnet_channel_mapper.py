from mmdet.models.builder import NECKS
from mmdet.models.necks import ChannelMapper


@NECKS.register_module()
class CBChannelMapper(ChannelMapper):

    def __init__(self, cb_idx=1, **kwargs):
        super(CBChannelMapper, self).__init__(**kwargs)
        self.cb_idx = cb_idx

    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]

        if self.training:
            outs = []
            # from IPython import embed; embed()
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[self.cb_idx])
            return out
