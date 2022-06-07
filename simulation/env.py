import os

from rllab.envs.normalized_env import normalize as normalize_env
from sandbox.rocky.tf.envs.base import TfEnv
from julia_env.julia_env import JuliaEnv
from hgail.envs.vectorized_normalized_env import vectorized_normalized_env
import hgail.misc.utils


def normalize_env_reset_with_kwargs(self, **kwargs):
    ret = self._wrapped_env.reset(**kwargs)
    if self._normalize_obs:
        return self._apply_normalize_obs(ret)
    else:
        return ret


def add_kwargs_to_reset(env):
    normalize_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalize_env is not None:
        normalize_env.reset = normalize_env_reset_with_kwargs.__get__(normalize_env)


def build_ngsim_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=True,
        render_params=None,
        videoMaking=False):
    basedir = os.path.expanduser(args.ngsim_data_dir)
    filepaths = [os.path.join(basedir, args.ngsim_filename)]
    if render_params is None:
        render_params = dict(
            viz_dir=os.path.join(exp_dir, 'imitate/viz'),
            zoom=5.
        )
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=args.env_H,
        primesteps=args.env_primesteps,
        action_repeat=args.env_action_repeat,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=render_params,
        n_envs=args.n_envs,
        n_veh=args.n_envs,
        remove_ngsim_veh=args.remove_ngsim_veh,
        reward=args.env_reward
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must
    # also be true

    if args.env_multiagent:
        env_id = 'MultiagentNGSIMEnv'
        if videoMaking:
            print('RAUNAK BHATTACHARRYA VIDEO MAKER IS ON')
            env_id='MultiagentNGSIMEnvVideoMaker'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env
    elif vectorize:
        env_id = 'VectorizedNGSIMEnv'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env

    else:
        env_id = 'NGSIMEnv'
        normalize_wrapper = normalize_env
    print(env_params)
    env = JuliaEnv(
        env_id=env_id,
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    add_kwargs_to_reset(env)
    return env, low, high