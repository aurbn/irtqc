#!/usr/bin/env python3
from argparse import ArgumentParser
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyteomics import mzml
#import lcmsms as lcmsms
#from irtqc import lcmsms
import lcmsms


if __name__ == '__main__':
    argparser = ArgumentParser(description="iRT peptide QC tool")
    argparser.add_argument('--mzml', type=str, required=True, help="MzML file")
    argparser.add_argument('--targets', type=str, required=True, help="Targets file")
    argparser.add_argument('--ms1-ppm', type=float, default=5, help="MS1 extraction window in ppm")
    argparser.add_argument('--ms2-prec-tolerance', type=float, default=0.01, help="MS2 precursor tolerance")
    argparser.add_argument('--ms2-frag-tolerance', type=float, default=1, help="MS2 precursor tolerance")
    argparser.add_argument('--width-1-pc', type=float, default=50, help="Chromatographic width 1 in %% of apex")
    argparser.add_argument('--width-2-pc', type=float, default=5, help="Chromatographic width 2 in %% of apex")
    argparser.add_argument('--debug', action="store_true", help="Pickle cache input file")
    argparser = argparser.parse_args()


    b_fname = ".".join(argparser.mzml.split(".")[:-1])
    pdf = PdfPages(b_fname+"_Figs.pdf")

    if argparser.debug:
        import pickle
        import time
        import os

        if os.path.exists(argparser.mzml+".pkl"):
            with open(argparser.mzml+".pkl", "rb") as f_:
                _start_time = time.time()
                print("Unpickling")
                exp = lcmsms.LCMSMSExperiment(tqdm.tqdm(pickle.load(f_)),
                                              prec_tolerance=argparser.ms2_prec_tolerance)
                print(f"Unpickled in {time.time()-_start_time} seconds")
        else:
            print("Reading and pickling")
            mzml_ = list(tqdm.tqdm(mzml.MzML(argparser.mzml)))
            with open(argparser.mzml+".pkl", "wb") as f_:
                pickle.dump(mzml_, f_)
            print("Pickled, parsing experiment")
            exp = lcmsms.LCMSMSExperiment(tqdm.tqdm(mzml_),
                                          prec_tolerance=argparser.ms2_prec_tolerance)
            del mzml_
    else:
        exp = lcmsms.LCMSMSExperiment(tqdm.tqdm(mzml.MzML(argparser.mzml)),
                                      prec_tolerance=argparser.ms2_prec_tolerance)

    ###  MS1 processing  ####
    targets = pd.read_csv(argparser.targets, sep='\t')
    targets_ms1 = targets[["Sequence", "Precursor_Mz"]].drop_duplicates()
    results_ms1 = pd.DataFrame(columns=["Sequence",
                                        "Precursor_Mz",
                                        "Apex_time",
                                        f"Width_{argparser.width_1_pc}_pc_time_start",
                                        f"Width_{argparser.width_1_pc}_pc_time_end",
                                        f"Width_{argparser.width_1_pc}_xic_area",
                                        f"Width_{argparser.width_2_pc}_pc_time_start",
                                        f"Width_{argparser.width_2_pc}_pc_time_end",
                                        f"Width_{argparser.width_2_pc}_xic_area",
                                        f"MS1_mass_apex_mz",
                                        f"MS1_apex_height",
                                        f"MS1_peak_halfwidth",
                                        f"MS1_peak_area",
                                        f"TIC_MS2",
                                        ])

    #from matplotlib.backends.backend_pdf import PdfPages
    #pdf = PdfPages('MS1.pdf')

    fig, axs = plt.subplots(len(targets_ms1), 4, figsize=(15, 60), gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.5)

    n=0

    for k, row in targets_ms1.iterrows():
        mz = row["Precursor_Mz"]
        seq = row["Sequence"]
        ch = exp.ms1.xic(mz, argparser.ms1_ppm)
        chs = ch.smooth(sigma=2)
        apext, apexi = ch.get_apex()
        width1 = chs.get_width_pc(argparser.width_1_pc)
        width2 = chs.get_width_pc(argparser.width_2_pc)
        area1 = chs.get_width_pc_area(argparser.width_1_pc)
        area2 = chs.get_width_pc_area(argparser.width_2_pc)
        spec = exp.ms1[apext]
        ms1_apex_mz, ms1_apex_int = spec.get_apex_around(mz, 0.05)
        ms1_hw = spec.get_apex_width_pc(mz, apex_pc=50, tolerance=0.05)
        ms1_area = spec.get_peak_area(mz, tolerance=0.05)

        ### PLOTS ###
        # XIC
        axs[n, 0].ticklabel_format(axis="y", style = 'sci', scilimits=(0,0))
        axs[n, 0].ticklabel_format(axis="x", style = 'plain')
        axs[n, 0].plot(ch.t, ch.i, "g-")
        #axs[n, 0].plot(xictimes, xic)
        #axs[n, 0].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 0].vlines(apext, 0, apexi * 1.1)
        axs[n, 0].title.set_text(f"{seq}\nmz={mz:.4f}\napex@{apext:.2f}min")
        axs[n, 0].set_xlim(15, 30)

        # XIC zoom
        axs[n, 1].ticklabel_format(axis="y", style = 'sci', scilimits=(0,0))
        axs[n, 1].ticklabel_format(axis="x", style = 'plain')
        axs[n, 1].plot(ch.t, ch.i, "gx-")
        axs[n, 1].plot(chs.t, chs.i, "rx-")
        #axs[n, 1].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 1].vlines(apext, 0, apexi)
        axs[n, 1].title.set_text(f"MS1 XIC zoon\n mz={mz:.4f}")
        axs[n, 1].hlines(apexi *0.5, *width1)
        axs[n, 1].hlines(apexi *0.05, *width2)
        axs[n, 1].set_xlim(apext - 0.2, apext + 0.4)
        axs[n, 1].text(0.45, 0.95, f"Area50={area1:.3e}\nArea5  ={area2:.3e}", transform=axs[n, 1].transAxes,
                       fontsize=10, verticalalignment='top')

        # MS1 spectrum
        spec = exp.ms1[apext]
        axs[n, 2].ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        axs[n, 2].ticklabel_format(axis="x", style='plain')
        axs[n, 2].title.set_text(f"MS1 spectrum\n@time={apext:.2f}min")
        spec.plot(ax=axs[n, 2], marks=[ms1_apex_mz])

        # MS1 spectrum zoom
        ms1_tolerance = ms1_apex_mz*argparser.ms1_ppm*1e-6
        spec_zoom = spec[ms1_apex_mz-argparser.ms2_prec_tolerance:ms1_apex_mz+argparser.ms2_prec_tolerance]  # No /2 (sic!)
        axs[n, 3].ticklabel_format(axis="y", style = 'sci', scilimits=(0,0))
        axs[n, 3].ticklabel_format(axis="x", style='sci', scilimits=(-3,-3), useOffset=ms1_apex_mz)
        #axs[n, 3].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        spec_zoom.plot("go-", ax=axs[n, 3])
        axs[n, 3].title.set_text("MS1 zoom\n mz={:.4f}".format(mz))
        axs[n, 3].vlines(ms1_apex_mz, 0, ms1_apex_int, "r")
        ms1_w_left, ms1_w_right = spec.get_apex_times_pc(ms1_apex_mz, apex_pc=50, tolerance=0.05)
        axs[n, 3].hlines(ms1_apex_int/2, ms1_w_left, ms1_w_right, "r")
        axs[n, 3].text(0.55, 0.95, f"Area={ms1_area:.2e}\nHW={ms1_hw:.2e}", transform=axs[n, 3].transAxes,
                       fontsize=10, verticalalignment='top')

        n+=1
        ############

        row['Apex_time'] = apext
        row[f"Width_{argparser.width_1_pc}_pc_time_start"] = width1[0]
        row[f"Width_{argparser.width_1_pc}_pc_time_end"] = width1[1]
        row[f"Width_{argparser.width_1_pc}_xic_area"] = area1
        row[f"Width_{argparser.width_2_pc}_pc_time_start"] = width2[0]
        row[f"Width_{argparser.width_2_pc}_pc_time_end"] = width2[1]
        row[f"Width_{argparser.width_2_pc}_xic_area"] = area2
        row[f"MS1_mass_apex_mz"] = ms1_apex_mz
        row[f"MS1_apex_height"] = ms1_apex_int
        row[f"MS1_peak_halfwidth"] = ms1_hw
        row[f"MS1_peak_area"] = ms1_area

        results_ms1 = results_ms1.append(row)

    pdf.savefig(fig)
    plt.close(fig)



    ###  MS2 processing  ###
    results_ms1.set_index("Sequence", drop=True, inplace=True)
    targets_ms2 = targets[["Sequence", "Precursor_Mz", "Product_Mz"]].drop_duplicates()
    results_ms2 = pd.DataFrame(columns=["Sequence",
                                        "Precursor_Mz",
                                        "Product_Mz",
                                        "MS2_TIC_Apex_time",
                                        "MS2_mass_apex_mz",
                                        "MS2_apex_height",
                                        "MS2_peak_halfwidth",
                                        "MS2_peak_area",
                               ])
    n_ = max(map(lambda x: len(x[1]), targets_ms2.groupby(by=["Sequence", "Precursor_Mz"])))
    fig, axs = plt.subplots(len(targets_ms1), 3+n_, figsize=(15, 100))
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(hspace=0.5)

    n = -1
    for k, grp in targets_ms2.groupby(by=["Sequence", "Precursor_Mz"]):
        n+=1
        seq = k[0]
        prec = k[1]
        apext = results_ms1.loc[seq, "Apex_time"]
        start = results_ms1.loc[seq, f"Width_{argparser.width_2_pc}_pc_time_start"]
        stop = results_ms1.loc[seq, f"Width_{argparser.width_2_pc}_pc_time_end"]

        ms2_all = exp.ms2.extract(prec)
        ms2_ext = ms2_all[start-argparser.ms2_frag_tolerance/2:stop+argparser.ms2_frag_tolerance/2]

        #spec = ms2_ext[apext]
        tic_apext, tic_apexint = ms2_ext.tic.get_apex()
        results_ms1.loc[seq, "TIC_MS2"] = tic_apexint
        spec = ms2_ext[tic_apext]

        ### PLOTS ###
        # TIC MS2
        axs[n, 0].ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        axs[n, 0].ticklabel_format(axis="x", style='plain')
        axs[n, 0].plot(ms2_all.tic.t, ms2_all.tic.i, "g-")
        axs[n, 0].title.set_text("TIC MS2\nmz={:.4f}\n apex@{:.2f}\n".format(prec, tic_apext))
        # TIC MS2 zoom
        axs[n, 1].ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        axs[n, 1].ticklabel_format(axis="x" , style='plain')
        axs[n, 1].plot(ms2_ext.tic.t, ms2_ext.tic.i, "g-")
        axs[n, 1].vlines(tic_apext, 0, tic_apexint, "r")
        axs[n, 1].title.set_text(f"TIC MS2 zoom\nmz={prec:.4f}\n apex@{tic_apext:.2f}\n")
        # MS2 spectrum
        axs[n, 2].ticklabel_format(axis="y", style='sci', scilimits=(0,0))
        axs[n, 2].ticklabel_format(axis="x", style='plain')
        spec.plot(ax=axs[n, 2])
        axs[n, 2].title.set_text(f"MS/MS for\n{prec:.4f}\n@time={tic_apext:.2f}min\n")

        nn = 3
        for kk, row in grp.iterrows():
            frag = row["Product_Mz"]
            try:
                fmz, fint = spec.get_apex_around(frag, argparser.ms2_frag_tolerance)
                f_hw = spec.get_apex_width_pc(frag, apex_pc=50, tolerance=argparser.ms2_frag_tolerance)
                f_area = spec.get_peak_area(fmz, tolerance=argparser.ms2_frag_tolerance)
                s_ext = spec[fmz-argparser.ms2_frag_tolerance/2:fmz+argparser.ms2_frag_tolerance/2]
                s_ext.plot("go-", ax=axs[n, nn])
                axs[n, nn].ticklabel_format(axis="y",style='sci', scilimits=(0, 0))
                #axs[n, nn].ticklabel_format(axis="x", style='sci', scilimits=(-2,-2), useOffset=fmz)
                axs[n, nn].ticklabel_format(axis="x", style='plain')
                axs[n, nn].title.set_text(f"MS2 zoom\nmz={fmz:.4f}\n")
                axs[n, nn].vlines(fmz, 0, max(s_ext.i), "r")
                axs[n, nn].vlines(frag, 0, max(s_ext.i), "blue")
                axs[n, 2].plot([frag], [fint], "rx")#, markersize=15)
                ms2_w_left, ms2_w_right = s_ext.get_apex_times_pc(fmz, apex_pc=50, tolerance=0.05)
                axs[n, nn].hlines(fint / 2, ms2_w_left, ms2_w_right, "r")
                axs[n, nn].text(0.6, 0.98, f"Area=\n{ms1_area:.2e}\n\nHW=\n{ms1_hw:.2e}", transform=axs[n, nn].transAxes,
                               fontsize=8, verticalalignment='top')


            except lcmsms.PeaksNotFound:
                print(f"No MS2 peak for {prec:.4f}/{frag:.4f}")
                f_hw = 0
                f_area = 0
                fmz = 0
                fint = 0
            nn += 1

            row["MS2_TIC_Apex_time"] = tic_apext
            row["MS2_mass_apex_mz"] = fmz
            row["MS2_apex_height"] = fint
            row["MS2_peak_halfwidth"] = f_hw
            row["MS2_peak_area"] = f_area
            results_ms2 = results_ms2.append(row)

    pdf.savefig(fig)
    plt.close(fig)

    results_ms2.set_index("Sequence", drop=True, inplace=True)
    fig.savefig("MS1.pdf", dpi=1200, format='pdf', bbox_inches='tight')
    ms1_fname = b_fname+"_MS1_table.csv"
    ms2_fname = b_fname+"_MS2_table.csv"
    results_ms1.to_csv(ms1_fname, sep='\t')
    results_ms2.to_csv(ms2_fname, sep='\t')

    pdf.close()



        

