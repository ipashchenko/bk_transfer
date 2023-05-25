import numpy as np
import argparse
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import modelfit_core_wo_extending


if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--fname",
                     type=str)
    CLI.add_argument("--beam_fractions",  # name on the CLI - drop the `--` for positional/required parameters
                    nargs="*",  # 0 or more values expected => creates a list
                    type=float,
                    default=[1.0],  # default if nothing is provided
                    )
    CLI.add_argument("--mapsize_clean",  # name on the CLI - drop the `--` for positional/required parameters
                     nargs="*",  # 0 or more values expected => creates a list
                     type=float,
                     default=[1024, 0.1],  # default if nothing is provided
                     )
    CLI.add_argument("--nw_beam_size",  # name on the CLI - drop the `--` for positional/required parameters
                     type=float
                     # default=[1.353, 1.557, -65.65],  # default if nothing is provided
                     # default=[4.561, 5.734, -51.67],  # default if nothing is provided
                     )
    CLI.add_argument("--use_elliptical",
                     type=bool)
    CLI.add_argument("--save_dir",
                     type=str)
    CLI.add_argument("--path_to_script",
                     type=str)
    args = CLI.parse_args()

    fname = args.fname
    beam_fractions = args.beam_fractions
    save_dir = args.save_dir
    nw_beam_size = args.nw_beam_size
    path_to_script = args.path_to_script
    use_elliptical = args.use_elliptical

    print("==================")
    print("Arguments parsed: ")
    print("use_elliptical = ", use_elliptical)


    results = modelfit_core_wo_extending(fname,
                                         beam_fractions,
                                         path=save_dir,
                                         mapsize_clean=args.mapsize_clean,
                                         path_to_script=path_to_script,
                                         niter=500,
                                         out_path=save_dir,
                                         use_brightest_pixel_as_initial_guess=True,
                                         estimate_rms=True,
                                         stokes="i",
                                         use_ell=use_elliptical, two_stage=False, use_scipy=False,
                                         nw_beam_size=nw_beam_size)

    # Flux of the core
    flux = np.mean([results[frac]['flux'] for frac in beam_fractions])
    rms_flux = np.std([results[frac]['flux'] for frac in beam_fractions])

    # Position of the core
    r = np.mean([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])
    rms_r = np.std([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])

    # Post-fit rms in the core region
    postfit_rms = np.median([results[frac]['rms'] for frac in beam_fractions])

    print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
    print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))
