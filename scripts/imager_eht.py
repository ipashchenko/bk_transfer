import numpy as np
import ehtim as eh
import matplotlib
matplotlib.use('Agg')


def image_in_eht(uvfits, prior_fits, maxit_first, maxit_iter,
                 max_number_of_cycles=1, frac_clipfloor=0.01, uvfits_beam=None,
                 frac_nchi2_to_stop=0.01, d1='vis', d2=False, d3=False,
                 alpha_d1=100, alpha_d2=100, alpha_d3=100,
                 s1='simple', s2=False, s3=False,
                 alpha_s1=1, alpha_s2=1, alpha_s3=1,
                 alpha_flux=500, alpha_cm=500, outname=None, pdfname=None, **eht_kwargs):
    """
    Simple script to image in eht-imaging library.

    :param clipfloor:
        Image pixels of prior image with intensity < clipfloor*PriorImage will
        be masked using embedding.

    From eht-imaging:
    d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
    d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
    d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'

    s1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'tv2', 'l1', 'patch','compact','compact2','rgauss'
    s2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'
    s3 (str): The third regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'

    alpha_d1 (float): The first data term weighting
    alpha_d2 (float): The second data term weighting
    alpha_d2 (float): The third data term weighting

    alpha_s1 (float): The first regularizer term weighting
    alpha_s2 (float): The second regularizer term weighting
    alpha_s3 (float): The third regularizer term weighting

    alpha_flux (float): The weighting for the total flux constraint
    alpha_cm (float): The weighting for the center of mass constraint
    """
    # Some big value to start with
    previous_chi2_normalized = 1000
    obs = eh.obsdata.load_uvfits(uvfits)
    # Fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    beamparams = obs.fit_beam(weighting="natural")
    # Nominal array resolution, 1/longest baseline
    res = obs.res()

    # Setting prior image
    imp = eh.image.load_fits(prior_fits, aipscc=False)
    # If aipscc=True => it uses delta pulse function. Hence fix it. Its OK for Prior image.
    imp.pulse = eh.trianglePulse2D

    # If data term include ``cphase``, ``chpase_diag`` or ``bs`` - shift image to make zero center of mass
    do_shift = False
    for d in (d1, d2, d3):
        if d in ("cphase", "cphase_diag", "bs"):
            do_shift = True
    if do_shift:
        print("Moving center of mass of the prior image to (0,0)")
        imp = imp.center()

    imp = imp.threshold(cutoff=frac_clipfloor, fill_val=0.0)

    chi2_normalized_cc = obs.chisq(imp, dtype=d1)
    if d2:
        chi2_normalized_cc += obs.chisq(imp, dtype=d2)
    if d3:
        chi2_normalized_cc += obs.chisq(imp, dtype=d3)
    print("nChi2 for CLEAN model is {:.3f}".format(chi2_normalized_cc))

    imp = imp.blur_circ(beamparams[0])
    fig = imp.display(scale="log", export_pdf="prior.pdf")

    # First iteration with CLEAN-image prior
    out = eh.imager_func(obs, imp, imp, imp.total_flux(),
                         d1, d2, d3, alpha_d1, alpha_d2, alpha_d3, s1, s2, s3, alpha_s1, alpha_s2, alpha_s3,
                         alpha_flux, alpha_cm, clipfloor=frac_clipfloor*np.max(imp.imvec),
                         maxit=maxit_first, ttype='nfft', weighting="natural", **eht_kwargs)

    current_chi2_normalized = obs.chisq(out, dtype=d1)
    if d2:
        current_chi2_normalized += obs.chisq(imp, dtype=d2)
    if d3:
        current_chi2_normalized += obs.chisq(imp, dtype=d3)
    print("First iteration nchi2 = {:.3f}".format(current_chi2_normalized))

    count = 0
    # Iterate until nchi2 doesn't change a lot
    while abs(previous_chi2_normalized - current_chi2_normalized) > frac_nchi2_to_stop*current_chi2_normalized:

        if count == max_number_of_cycles:
            break

        out = out.blur_circ(res)
        out = eh.imager_func(obs, out, out, imp.total_flux(),
                             d1, d2, d3, alpha_d1, alpha_d2, alpha_d3, s1, s2, s3, 0.3*alpha_s1, alpha_s2, alpha_s3,
                             alpha_flux, alpha_cm, clipfloor=frac_clipfloor*np.max(out.imvec),
                             maxit=maxit_iter, ttype='nfft', weighting="natural", **eht_kwargs)
        previous_chi2_normalized = current_chi2_normalized
        current_chi2_normalized = obs.chisq(out, dtype=d1)
        if d2:
            current_chi2_normalized += obs.chisq(imp, dtype=d2)
        if d3:
            current_chi2_normalized += obs.chisq(imp, dtype=d3)
        print("Next iteration nchi2 = {:.3f}".format(current_chi2_normalized))
        count += 1

    if uvfits_beam is not None:
        obs = eh.obsdata.load_uvfits(uvfits_beam)
        beamparams = obs.fit_beam("natural")
    outblur = out.blur_gauss(beamparams, 1.0)

    # Plot the image
    if pdfname is not None:
        fig = outblur.display(scale="log", export_pdf=pdfname)
    # Save the images (both beam-convolved and original MEM)
    if outname is not None:
        out.save_txt(outname + '.txt')
        out.save_fits(outname + '.fits')
        outblur.save_txt(outname + '_blur.txt')
        outblur.save_fits(outname + '_blur.fits')

    return out


if __name__ == "__main__":
    uvfits_high = "/home/ilya/data/alpha/results/MOJAVE/2ridges/template_15.4.uvf"
    uvfits_low = "/home/ilya/data/alpha/results/MOJAVE/2ridges/template_8.1.uvf"
    initial_fits_high = "/home/ilya/data/alpha/results/MOJAVE/2ridges/convolved_true_jet_model_i_rotated_15.4.fits"
    initial_fits_low = "/home/ilya/data/alpha/results/MOJAVE/2ridges/convolved_true_jet_model_i_rotated_8.1.fits"

    eht_image = image_in_eht(uvfits_high, initial_fits_high, uvfits_beam=uvfits_low,
                             frac_clipfloor=0.00001,
                             maxit_first=500, maxit_iter=800, frac_nchi2_to_stop=0.03,
                             alpha_flux=0.0, alpha_cm=0.0,
                             d1="vis", alpha_d1=1,
                             s1="simple", s2="tv", alpha_s1=10.0, alpha_s2=1.0,
                             outname="mojave", pdfname="mojave.pdf")


    import sys
    sys.exit(0)
    sys.path.insert(0, '/home/ilya/github/ve/vlbi_errors')
    from image import plot as iplot
    from from_fits import create_image_from_fits_file
    import astropy.io.fits as pf
    import astropy.units as u

    obs = eh.obsdata.load_uvfits(uvfits_low)
    beamparams = obs.fit_beam(weighting="natural")
    im8 = pf.getdata("/home/ilya/github/bk_transfer/scripts/bk145_8.1_blur.fits")
    im15 = pf.getdata("/home/ilya/github/bk_transfer/scripts/bk145_15.4_blur.fits")
    alpha = np.log(im8/im15)/np.log(8.1/15.4)
    im8_clean = create_image_from_fits_file("/home/ilya/data/alpha/results/BK145/bk/model_cc_i_8.1.fits")
    beam = (beamparams[0]*u.rad.to(u.mas), beamparams[1]*u.rad.to(u.mas), beamparams[2])
    fig = iplot(im8, alpha, x=im8_clean.x, y=im8_clean.y,
                colors_mask=im8 < 0.0005*np.max(im8),
                color_clim=[-1.5, 0.5], colorbar_label=r"$\alpha$",
                min_abs_level=0.0005*np.max(im8), blc=(450, 440), trc=(900, 685),
                beam=beam, close=False, show_beam=True, show=True,
                contour_color='black', contour_linewidth=0.25, cmap="bwr")
