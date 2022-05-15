###########################
# Latent ODEs for Irregularly-Sampled Time Series
# https://github.com/YuliaRubanova/latent_ode/blob/master/lib/latent_ode.py
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn

from .modules import utils
from .modules.utils import get_device
from .modules.encoder_decoder import Encoder_z0_ODE_RNN, Encoder_z0_RNN, Decoder
from .modules.base_vae import VAE_Baseline
from .modules.diffeq_solver import DiffeqSolver
from .modules.ode_func import ODEFunc, ODEFunc_w_Poisson


class LatentODE(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
                 z0_prior, device, obsrv_std=None,
                 use_binary_classif=False, use_poisson_proc=False,
                 linear_classifier=False,
                 classif_per_tp=False,
                 n_labels=1,
                 train_classif_w_reconstr=False):

        super(LatentODE, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device, obsrv_std=obsrv_std,
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            linear_classifier=linear_classifier,
            use_poisson_proc=use_poisson_proc,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps,
                           mask=None, n_traj_samples=1, run_backwards=True):

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
                isinstance(self.encoder_z0, Encoder_z0_RNN):

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards=run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))

        first_point_std = first_point_std.abs()
        assert (torch.sum(first_point_std < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj, self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())
        assert (not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert (torch.sum(int_lambda[:, :, 0, :]) == 0.)
            assert (torch.sum(int_lambda[0, 0, -1, :] <= 0) == 0.)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            # intergral of lambda from the last step of ODE Solver
            all_extra_info["int_lambda"] = int_lambda[:, :, -1, :]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1):
        # input_dim = starting_point.size()[-1]
        # starting_point = starting_point.view(1,1,input_dim)

        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = starting_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros(n_traj_samples, n_traj, self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict,
                                                          n_traj_samples=3)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

        return self.decoder(sol_y)


def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
                           classif_per_tp=False, n_labels=1):
    # dim = args.latents
    # if args.poisson:
    #     lambda_net = utils.create_net(dim, input_dim,
    #                                   n_layers=1, n_units=args.units, nonlinear=nn.Tanh)
    #
    #     # ODE function produces the gradient for latent state and for poisson rate
    #     ode_func_net = utils.create_net(dim * 2, args.latents * 2,
    #                                     n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)
    #
    #     gen_ode_func = ODEFunc_w_Poisson(
    #         input_dim=input_dim,
    #         latent_dim=args.latents * 2,
    #         ode_func_net=ode_func_net,
    #         lambda_net=lambda_net,
    #         device=device).to(device)
    # else:
    dim = args.latents
    ode_func_net = utils.create_net(dim, args.latents,
                                    n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)

    gen_ode_func = ODEFunc(
        input_dim=input_dim,
        latent_dim=args.latents,
        ode_func_net=ode_func_net,
        device=device).to(device)

    z0_diffeq_solver = None
    n_rec_dims = args.rec_dims
    enc_input_dim = int(input_dim) * 2  # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = args.latents
    # if args.poisson:
    #     z0_dim += args.latents  # predict the initial poisson rate

    if args.z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
                                        n_layers=args.rec_layers, n_units=args.units, nonlinear=nn.Tanh)

        rec_ode_func = ODEFunc(
            input_dim=enc_input_dim,
            latent_dim=n_rec_dims,
            ode_func_net=ode_func_net,
            device=device).to(device)

        z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents,
                                        odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

        encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
                                        z0_dim=z0_dim, n_gru_units=args.gru_units, device=device).to(device)

    elif args.z0_encoder == "rnn":
        encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
                                    lstm_output_size=n_rec_dims, device=device).to(device)
    else:
        raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

    decoder = Decoder(args.latents, args.output_dim).to(device)

    diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents,
                                 odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

    model = LatentODE(
        input_dim=gen_data_dim,
        latent_dim=args.latents,
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=obsrv_std,
        use_poisson_proc=False, #args.poisson,
        use_binary_classif=False, #args.classif,
        linear_classifier=False, #args.linear_classif,
        classif_per_tp=classif_per_tp,
        n_labels=n_labels,
        # train_classif_w_reconstr=(args.dataset == "physionet")
    ).to(device)

    return model