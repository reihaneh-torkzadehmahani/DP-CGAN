# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from differential_privacy.optimizer import gaussian_query


def make_optimizer_class(cls):
    """Constructs a DP optimizer class from an existing one."""
    if (tf.train.Optimizer.compute_gradients.__code__ is
            not cls.compute_gradients.__code__):
        tf.logging.warning(
            'WARNING: Calling make_optimizer_class() on class %s that overrides '
            'method compute_gradients(). Check to ensure that '
            'make_optimizer_class() does not interfere with overridden version.',
            cls.__name__)

    class DPOptimizerClass(cls):
        """Differentially private subclass of given class cls."""

        def __init__(
                self,
                moment_accountant,
                l2_norm_clip,
                noise_multiplier,
                dp_average_query,
                num_microbatches,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self._accountant = moment_accountant
            self._dp_average_query = dp_average_query
            self._num_microbatches = num_microbatches
            self._global_state = self._dp_average_query.initial_global_state()

            # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
            # Beware: When num_microbatches is large (>100), enabling this parameter
            # may cause an OOM error.
            self._unroll_microbatches = unroll_microbatches

        def dp_compute_gradients(
                self,
                loss,
                var_list,
                gate_gradients=tf.train.Optimizer.GATE_OP,
                aggregation_method=None,
                colocate_gradients_with_ops=False,
                grad_loss=None, noise_flag = True):

            # Note: it would be closer to the correct i.i.d. sampling of records if
            # we sampled each microbatch from the appropriate binomial distribution,
            # although that still wouldn't be quite correct because it would be
            # sampling from the dataset without replacement.

            microbatches_losses = tf.reshape(loss, [self._num_microbatches, -1])
            sample_params = (self._dp_average_query.derive_sample_params(self._global_state))

            def process_microbatch(i, sample_state):

                """Process one microbatch (record) with privacy helper."""
                grads, _ = zip(*super(cls, self).compute_gradients(
                    tf.gather(microbatches_losses, [i]), var_list, gate_gradients,
                    aggregation_method, colocate_gradients_with_ops, grad_loss))

                # Converts tensor to list to replace None gradients with zero

                grads_list = list(grads)
                for inx in range(0, len(grads)):
                    if (grads[inx] == None):
                        grads_list[inx] = tf.zeros_like(var_list[inx])

                # update accountant

                if (noise_flag):
                    for grad in grads_list:
                        num_examples = tf.slice(tf.shape(grad), [0], [1])
                        privacy_accum_op = self.acc.accumulate_privacy_spending(
                            [None, None], self.noise_multiplier, num_examples)

                    # accumulate spent privacy budget

                    with tf.control_dependencies([privacy_accum_op]):

                        sample_state = self._dp_average_query.accumulate_record(
                            sample_params, sample_state, grads_list)

                else:

                    sample_state = self._dp_average_query.accumulate_record(
                        sample_params, sample_state, grads_list)

                return sample_state

            if var_list is None:
                var_list = (
                        tf.trainable_variables() + tf.get_collection(
                    tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
            sample_state = self._dp_average_query.initial_sample_state(
                self._global_state, var_list)

            if self._unroll_microbatches:
                for idx in range(self._num_microbatches):
                    sample_state = process_microbatch(idx, sample_state)
            else:
                # Use of while_loop here requires that sample_state be a nested
                # structure of tensors. In general, we would prefer to allow it to be
                # an arbitrary opaque type.

                cond_fn = lambda i, _: tf.less(i, self._num_microbatches)
                body_fn = lambda i, state: [tf.add(i, 1), process_microbatch(i, state)]

                idx = tf.constant(0)
                _, sample_state = tf.while_loop(cond_fn, body_fn, [idx, sample_state])

            final_grads, self._global_state = (
                self._dp_average_query.get_noised_result(
                    sample_state, self._global_state, noise_flag))

            return (final_grads)



        def minimize(
                self,
                d_loss_real,
                d_loss_fake,
                global_step=None,
                var_list=None,
                gate_gradients=tf.train.Optimizer.GATE_OP,
                aggregation_method=None,
                colocate_gradients_with_ops=False,
                name=None,
                grad_loss=None):

            """Minimize using sanitized gradients
            Add up gradient of loss on real data and fake data
            Computes gradient of this loss ,clip it and make it noisy,
            apply gradients


            Args:
              d_loss_real: the loss tensor for real data
              d_loss_fake: the loss tensor for fake data
              global_step: the optional global step.
              var_list: the optional variables.
              name: the optional name.
            Returns:
              the operation that runs one step of DP gradient descent.
            """

            # First validate the var_list

            if var_list is None:
                var_list = tf.trainable_variables()
            for var in var_list:
                if not isinstance(var, tf.Variable):
                    raise TypeError("Argument is not a variable.Variable: %s" % var)

            # Modification: apply gradient once every batches_per_lot many steps.
            # This may lead to smaller error
            d_loss = d_loss_real + d_loss_fake
            s_grads = self.dp_compute_gradients(
                d_loss, var_list=var_list, gate_gradients=gate_gradients,
                aggregation_method=aggregation_method,
                colocate_gradients_with_ops=colocate_gradients_with_ops,
                grad_loss=grad_loss, noise_flag=True)


            sanitized_grads_and_vars = list(zip(s_grads, var_list))
            self._assert_valid_dtypes([v for g, v in sanitized_grads_and_vars if g is not None])

            # Apply the overall gradients

            apply_grads = self.apply_gradients(sanitized_grads_and_vars, global_step=global_step, name=name)

            return apply_grads

    return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
    """Constructs a DP optimizer with Gaussian averaging of updates."""

    class DPGaussianOptimizerClass(make_optimizer_class(cls)):
        """DP subclass of given class cls using Gaussian averaging."""

        def __init__(
                self,
                moment_accountant,
                l2_norm_clip,
                noise_multiplier,
                num_microbatches,
                unroll_microbatches=False,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs):
            dp_average_query = gaussian_query.GaussianAverageQuery(
                l2_norm_clip, l2_norm_clip * noise_multiplier, num_microbatches)
            self.acc = moment_accountant
            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier

            super(DPGaussianOptimizerClass, self).__init__(
                moment_accountant,
                l2_norm_clip,
                noise_multiplier,
                dp_average_query,
                num_microbatches,
                unroll_microbatches,
                *args,
                **kwargs)

    return DPGaussianOptimizerClass


DPAdagradOptimizer = make_optimizer_class(tf.train.AdagradOptimizer)
DPAdamOptimizer = make_optimizer_class(tf.train.AdamOptimizer)
DPGradientDescentOptimizer = make_optimizer_class(
    tf.train.GradientDescentOptimizer)

DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(
    tf.train.AdagradOptimizer)
DPAdamGaussianOptimizer = make_gaussian_optimizer_class(tf.train.AdamOptimizer)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(
    tf.train.GradientDescentOptimizer)

