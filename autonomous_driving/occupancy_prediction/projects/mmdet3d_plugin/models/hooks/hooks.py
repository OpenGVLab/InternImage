from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad is None and val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))
