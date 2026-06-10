from learning_rules.eprop      import (compute_eprop_gradients,
                                        compute_eprop_leaky_gradients,
                                        mse_error, xent_error)
from learning_rules.deep_eprop import compute_deep_eprop_gradients
from learning_rules.deep_rtrl  import compute_deep_rtrl_gradients
from learning_rules.bptt       import compute_bptt_gradients, _mse_loss, _xent_loss
from learning_rules.interface  import (LearningRule, make_learning_rule,
                                        apply_gradients, lr_for_config)
