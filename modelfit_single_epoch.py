import os
import numpy as np
import json
from tempfile import TemporaryDirectory
import shutil
import argparse
from pprint import pprint
import astropy.units as u
deg2rad = u.deg.to(u.rad)
import sys
sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
from spydiff import clean_difmap, find_nw_beam, modelfit_difmap
from from_fits import create_clean_image_from_fits_file
from uv_data import UVData


def filter_difmap_CC_model_by_r(old_difmap_mdl_file, new_difmap_model_file,
                                center, r_min_mas):
    new_lines = list()
    with open(old_difmap_mdl_file, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            if line.startswith("!"):
                new_lines.append(line)
                continue
            flux, r, theta = line.split()
            flux = float(flux)
            r = float(r)
            theta = np.deg2rad(float(theta))
            ra = r*np.sin(theta)
            dec = r*np.cos(theta)
            # print("RA = {:.2f}, DEC = {:.2f}".format(ra, dec))
            if np.hypot(ra - center[0], dec - center[1]) > r_min_mas:
                new_lines.append(line)

    with open(new_difmap_model_file, "w") as fo:
        for line in new_lines:
            fo.write(line)

def join_difmap_models(dfm_model_1, dfm_model_2, out_file):
    """
    Join difmap VLBI models.
    """
    added_lines = list()

    with open(dfm_model_1, "r") as fo:
        lines = fo.readlines()
        added_lines = lines

    with open(dfm_model_2, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            if line.startswith("!"):
                continue
            added_lines.append(line)

    with open(out_file, "w") as fo:
        for line in added_lines:
            fo.write(line)


def create_difmap_file_from_single_component(comp, out_fname, freq_hz):
    """
    :param comp:
        (flux, ra, dec[, bmaj, e, bpa]) - [Jy, mas, mas, mas, -, deg]

    """

    with open(out_fname, "w") as fo:
        if len(comp) == 3:
            flux, ra, dec = comp
            e = "1.00000"
            bmaj = "0.0000"
            bpa = "000.000"
            type_ = "0"
        elif len(comp) == 4:
            flux, ra, dec, bmaj = comp
            e = "1.00000"
            bpa = "000.000"
            type_ = "1"
            bmaj = "{:.5f}v".format(bmaj)
        elif len(comp) == 6:
            flux, ra, dec, bmaj, e, bpa = comp
            e = "{:.5f}v".format(e)
            bpa = "{:.5f}v".format((bpa - np.pi / 2) / deg2rad)
            bmaj = "{:.5f}v".format(bmaj)
            type_ = "1"
        else:
            raise Exception
        # mas
        r = np.hypot(ra, dec)
        # rad
        theta = np.rad2deg(np.arctan2(ra, dec))
        fo.write("! Flux (Jy) Radius (mas)  Theta (deg)  Major (mas)  Axial ratio   Phi (deg) T\n\
! Freq (Hz)     SpecIndex\n")
        fo.write("{:.5f}v {:.5f}v {:.5f}v {} {} {} {} {:.2E} 0\n".format(flux, r, theta,
                                                         bmaj, e, bpa, type_,
                                                         freq_hz))

def modelfit_core_wo_extending(uvfits, mapsize_clean, beam_fractions, path_to_script, use_elliptical=False,
                               use_brightest_pixel_as_initial_guess=True, save_dir=None,
                               dump_json_result=True):


    result = dict()

    with TemporaryDirectory() as working_dir:
        # First CLEAN and dump difmap model file with CCs
        clean_difmap(uvfits, os.path.join(working_dir, "test_cc.fits"), "i",
                     mapsize_clean,
                     path_to_script=path_to_script,
                     mapsize_restore=None, beam_restore=None, shift=None,
                     show_difmap_output=True, command_file=None, clean_box=None,
                     save_dfm_model=os.path.join(working_dir, "cc.mdl"),
                     omit_residuals=False, do_smooth=True, dmap=None,
                     text_box=None, box_rms_factor=None, window_file=None,
                     super_unif_dynam=None, unif_dynam=None,
                     taper_gaussian_value=None, taper_gaussian_radius=None)

        uvdata = UVData(uvfits)
        freq_hz = uvdata.frequency
        # Find beam
        bmin, bmaj, bpa = find_nw_beam(uvfits, stokes="i", mapsize=mapsize_clean, uv_range=None, working_dir=working_dir)
        print("NW beam : {:.2f} mas, {:.2f} mas, {:.2f} deg".format(bmaj, bmin, bpa))

        # Find the brightest pixel
        ccimage = create_clean_image_from_fits_file(os.path.join(working_dir, "test_cc.fits"))
        if use_brightest_pixel_as_initial_guess:
            im = np.unravel_index(np.argmax(ccimage.image), ccimage.image.shape)
            print("indexes of max intensity ", im)
            # - to RA cause dx_RA < 0
            r_c = (-(im[1]-mapsize_clean[0]/2)*mapsize_clean[1],
                   (im[0]-mapsize_clean[0]/2)*mapsize_clean[1])
        else:
            r_c = (0, 0)
        print("Brightest pixel coordinates (RA, DEC) : {:.2f}, {:.2f}".format(r_c[0], r_c[1]))

        # Create model with a single component
        if not use_elliptical:
            comp = (1., r_c[0], r_c[1], 0.25)
        else:
            comp = (1., r_c[0], r_c[1], 0.25, 1.0, 0.0)
        create_difmap_file_from_single_component(comp, os.path.join(working_dir, "1.mdl"), freq_hz)

        for beam_fraction in beam_fractions:
            # Filter CCs
            filter_difmap_CC_model_by_r(os.path.join(working_dir, "cc.mdl"),
                                        os.path.join(working_dir, "filtered_cc.mdl"),
                                        r_c, bmaj*beam_fraction)
            # Add single gaussian component model to CC model
            join_difmap_models(os.path.join(working_dir, "filtered_cc.mdl"),
                               os.path.join(working_dir, "1.mdl"),
                               os.path.join(working_dir, "hybrid.mdl"))
            modelfit_difmap(uvfits, mdl_fname=os.path.join(working_dir, "hybrid.mdl"),
                            out_fname=os.path.join(working_dir, "hybrid_fitted.mdl"),
                            niter=100, stokes='i', show_difmap_output=True)


            # DEBUG
            shutil.copy(os.path.join(working_dir, "hybrid_fitted.mdl"), os.path.join(save_dir, "hybrid_fitted.mdl"))
            

            # Extract core parameters
            with open(os.path.join(working_dir, "hybrid_fitted.mdl"), "r") as fo:
                lines = fo.readlines()
                components = list()
                for line in lines:
                    if line.startswith("!"):
                        continue
                    splitted = line.split()
                    if len(splitted) == 3:
                        continue
                    if len(splitted) == 9:
                        flux, r, theta, major, axial, phi, type_, freq, spec  = splitted
                        flux = float(flux.strip("v"))
                        r = float(r.strip("v"))
                        theta = float(theta.strip("v"))
                        major = float(major.strip("v"))
                        axial = float(axial.strip("v"))
                        phi = float(phi.strip("v"))

                        theta = np.deg2rad(theta)
                        ra = r*np.sin(theta)
                        dec = r*np.cos(theta)


                        # CG
                        if type_ == "1":
                            component = (flux, ra, dec, major)
                        elif type_ == "2":
                            component = (flux, ra, dec, major, axial, phi)
                        else:
                            raise Exception("Component must be Circualr or Elliptical Gaussian!")
                        components.append(component)
                if len(components) > 1:
                    raise Exception("There should be only one core component!")
                if not components:
                    raise Exception("No core component found!")
                # return components[0]

                if not use_elliptical:
                    flux, ra, dec, size = components[0]
                    result.update({beam_fraction: {"flux": flux, "ra": ra,
                                                   "dec": dec, "size": size,
                                                   "rms": np.nan}})
                else:
                    flux, ra, dec, size, e, bpa = components[0]
                    result.update({beam_fraction: {"flux": flux, "ra": ra,
                                                   "dec": dec, "size": size,
                                                   "e": e, "bpa": bpa,
                                                   "rms": np.nan}})
        # if save_dir is not None: 
        #     for fn in (os.path.join(working_dir, "filtered_cc.mdl"),
        #                os.path.join(working_dir, "1.mdl"),
        #                os.path.join(working_dir, "test_cc.fits"),
        #                os.path.join(working_dir, "hybrid.mdl"),
        #                os.path.join(working_dir, "hybrid_fitted.mdl")):
        #         fname = os.path.split(fn)[-1]
        #         shutil.move(fn, os.path.join(save_dir, fname))

    base = os.path.split(uvfits)[-1]
    base = base.split(".")[:-1]
    base = ".".join(base)
    if dump_json_result:
        with open(os.path.join(save_dir, f"{base}_core_modelfit_result.json"), "w") as fo:
            json.dump(result, fo)

    return result


if __name__ == "__main__":

    # uvfits = "/home/ilya/github/bk_transfer/pics/flares/tmp/template_S_1800.0.uvf"
    # result = modelfit_core_wo_extending(uvfits, mapsize_clean=(512, 0.5), beam_fractions=np.round(np.linspace(0.5, 1.5, 11), 2),
    #                                     path_to_script="/home/ilya/github/bk_transfer/scripts/script_clean_rms",
    #                                     use_elliptical=False,
    #                                     use_brightest_pixel_as_initial_guess=True,
    #                                     save_dir="/home/ilya/github/bk_transfer/pics/flares/tmp")
    # pprint(result)


    CLI = argparse.ArgumentParser()
    CLI.add_argument("--fname",
                     type=str)
    CLI.add_argument("--beam_fractions",  # name on the CLI - drop the `--` for positional/required parameters
                    nargs="*",  # 0 or more values expected => creates a list
                    type=float,
                    default=1.0,  # default if nothing is provided
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
                     type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    CLI.add_argument("--save_dir",
                     type=str)
    CLI.add_argument("--path_to_script",
                     type=str)
    args = CLI.parse_args()

    fname = args.fname
    save_dir = args.save_dir
    nw_beam_size = args.nw_beam_size
    path_to_script = args.path_to_script
    use_elliptical = args.use_elliptical
    mapsize_clean=args.mapsize_clean
    beam_fractions=args.beam_fractions

    print("==================")
    print("Arguments parsed: ")
    print("use_elliptical = ", use_elliptical)
    print("beam_fractions = ", beam_fractions)


    # results = modelfit_core_wo_extending(fname,
    #                                      beam_fractions,
    #                                      path=save_dir,
    #                                      mapsize_clean=args.mapsize_clean,
    #                                      path_to_script=path_to_script,
    #                                      niter=500,
    #                                      out_path=save_dir,
    #                                      use_brightest_pixel_as_initial_guess=True,
    #                                      estimate_rms=True,
    #                                      stokes="i",
    #                                      use_ell=use_elliptical, two_stage=False, use_scipy=False,
    #                                      nw_beam_size=nw_beam_size)

    fname = os.path.join(save_dir, fname)
    results = modelfit_core_wo_extending(fname, mapsize_clean=mapsize_clean, beam_fractions=beam_fractions,
                                         path_to_script=path_to_script,
                                         use_elliptical=use_elliptical,
                                         use_brightest_pixel_as_initial_guess=True,
                                         save_dir=save_dir, dump_json_result=True)
    # Flux of the core
    flux = np.mean([results[frac]['flux'] for frac in beam_fractions])
    rms_flux = np.std([results[frac]['flux'] for frac in beam_fractions])

    # Position of the core
    r = np.mean([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])
    rms_r = np.std([np.hypot(results[frac]['ra'], results[frac]['dec']) for frac in beam_fractions])

    # Post-fit rms in the core region
    postfit_rms = np.nanmedian([results[frac]['rms'] for frac in beam_fractions])

    print("Core flux : {:.2f}+/-{:.2f} Jy".format(flux, rms_flux))
    print("Core position : {:.2f}+/-{:.2f} mas".format(r, rms_r))
