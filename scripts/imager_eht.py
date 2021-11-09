import ehtim as eh
import matplotlib
matplotlib.use('Agg')


def image_in_eht(uvfits, prior_clean_fits, maxit, frac_nchi2_to_stop=0.01, d1='vis', d2=False, d3=False,
                 alpha_d1=100, alpha_d2=100, alpha_d3=100,
                 s1='simple', s2=False, s3=False,
                 alpha_s1=1, alpha_s2=1, alpha_s3=1,
                 alpha_flux=500, alpha_cm=500, outname=None, pdfname=None, **eht_kwargs):
    """
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
    beamparams = obs.fit_beam()
    # Nominal array resolution, 1/longest baseline
    res = obs.res()

    # Setting prior image
    # Beam-convolved CLEAN image as prior
    imp = eh.image.load_fits(prior_clean_fits, aipscc=True)
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

    imp = imp.threshold(cutoff=0.01)

    chi2_normalized_cc = obs.chisq(imp, dtype=d1)
    if d2:
        chi2_normalized_cc += obs.chisq(imp, dtype=d2)
    if d3:
        chi2_normalized_cc += obs.chisq(imp, dtype=d3)
    print("nChi2 for CLEAN model is {:.3f}".format(chi2_normalized_cc))

    # If aipscc=True => need to convolve with beam
    imp = imp.blur_circ(beamparams[0])

    # First iteration with CLEAN-image prior
    out = eh.imager_func(obs, imp, imp, imp.total_flux(),
                         d1, d2, d3, alpha_d1, alpha_d2, alpha_d3, s1, s2, s3, alpha_s1, alpha_s2, alpha_s3,
                         alpha_flux, alpha_cm,
                         maxit=maxit, ttype='nfft', weighting="natural", **eht_kwargs)

    current_chi2_normalized = obs.chisq(out, dtype=d1)
    if d2:
        current_chi2_normalized += obs.chisq(imp, dtype=d2)
    if d3:
        current_chi2_normalized += obs.chisq(imp, dtype=d3)
    print("First iteration nchi2 = {:.3f}".format(current_chi2_normalized))

    # Iterate until nchi2 doesn't change a lot
    while abs(previous_chi2_normalized - current_chi2_normalized) > frac_nchi2_to_stop*current_chi2_normalized:

        out = out.blur_circ(res)
        out = eh.imager_func(obs, out, out, imp.total_flux(),
                             d1, d2, d3, alpha_d1, alpha_d2, alpha_d3, s1, s2, s3, alpha_s1, alpha_s2, alpha_s3,
                             alpha_flux, alpha_cm,
                             maxit=maxit, ttype='nfft', weighting="natural", **eht_kwargs)
        previous_chi2_normalized = current_chi2_normalized
        current_chi2_normalized = obs.chisq(out, dtype=d1)
        if d2:
            current_chi2_normalized += obs.chisq(imp, dtype=d2)
        if d3:
            current_chi2_normalized += obs.chisq(imp, dtype=d3)
        print("Next iteration nchi2 = {:.3f}".format(current_chi2_normalized))

    outblur = out.blur_gauss(beamparams, 0.5)
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
    uvfits = "/home/ilya/data/alpha/MOJAVE/1228+126.u.2020_07_02.uvf"
    clean_image_tap = "/home/ilya/data/alpha/MOJAVE/1228+126.u.2020_07_02.ict.fits"
    eht_image = image_in_eht(uvfits, clean_image_tap, maxit=300, frac_nchi2_to_stop=0.03,
                             d1="vis", alpha_d1=1,
                             s1="simple", alpha_s1=1, outname="mojave", pdfname="mojave.pdf")
