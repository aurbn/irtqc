from argparse import ArgumentParser
import tqdm
import pandas as pd
import lcmsms
from matplotlib import pyplot as plt

if __name__ == '__main__':
    argparser = ArgumentParser(description="iRT peptide QC tool")
    argparser.add_argument('--mzml', type=str, required=True, help="MzML file")
    argparser.add_argument('--targets', type=str, required=True, help="Targets file")
    argparser.add_argument('--ms1-ppm', type=float, default=5, help="MS1 extraction window in ppm")
    argparser.add_argument('--ms2-prec-tolerance', type=float, default=0.01, help="MS2 precursor tolerance")
    argparser.add_argument('--width-1-pc', type=float, default=50, help="Cromatographic width 1 in % of apex")
    argparser.add_argument('--width-2-pc', type=float, default=5, help="Cromatographic width 2 in % of apex")
    argparser = argparser.parse_args()


    ##### FOR TESTING ####
    import pickle
    import time


    #exp = LCMSMSExperiment(tqdm.tqdm(mzml.MzML(argparser.mzml)))

    # mzml_ = list(tqdm.tqdm(mzml.MzML(argparser.mzml)))
    # with open("mzml_.pkl", "wb") as f_:
    #     pickle.dump(mzml_, f_)


    _start_time = time.time()
    with open("mzml_.pkl", "rb") as f_:
        print("Unpickling")
        exp = lcmsms.LCMSMSExperiment(tqdm.tqdm(pickle.load(f_)),
                                      prec_tolerance=argparser.ms2_prec_tolerance)
        print(f"Unpickled in {time.time()-_start_time} seconds")
    ### ####


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
                                        ])

    #from matplotlib.backends.backend_pdf import PdfPages
    #pdf = PdfPages('MS1.pdf')
    fig, axs = plt.subplots(len(targets_ms1), 2, figsize=(15, 40), gridspec_kw={'width_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.5)

    n=0

    for k, row in targets_ms1.iterrows():
        mz = row["Precursor_Mz"]
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

        ### PLOT ###
        axs[n, 0].plot(ch.t, ch.i, "g-")
        #axs[n, 0].plot(xictimes, xic)
        #axs[n, 0].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 0].vlines(apext, 0, apexi * 1.1)
        axs[n, 0].title.set_text("{}  mz={}  apex@{}".format(n, mz, apext))
        axs[n, 0].set_xlim(15, 30)

        axs[n, 1].plot(ch.t, ch.i, "gx-")
        axs[n, 1].plot(chs.t, chs.i, "rx-")
        #axs[n, 1].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 1].vlines(apext, 0, apexi)
        axs[n, 1].title.set_text("{}  mz={:.4f}".format(n, mz))
        axs[n, 1].hlines(apexi *0.5, *width1)#, xictimes[right1])
        axs[n, 1].hlines(apexi *0.05, *width2)# xictimes[left2], xictimes[right2])
        axs[n, 1].set_xlim(apext - 0.2, apext + 0.4)
        axs[n, 1].text(0.45, 0.95, f"Area50={area1:.3e}\nArea5  ={area2:.3e}", transform=axs[n, 1].transAxes,
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

    fig.savefig("MS1.pdf", dpi=1200, format='pdf', bbox_inches='tight')
    results_ms1.to_csv("MS1_test.csv", sep='\t', index=False)

    ###  MS2 processing  ###
    results_ms1.set_index("Sequence", drop=True, inplace=True)
    targets_ms2 = targets[["Sequence", "Precursor_Mz", "Product_Mz"]].drop_duplicates()
    results_ms2 = pd.DataFrame(columns=["Sequence",
                                        "Precursor_Mz",
                                        "Product_Mz",
                                        "Apex_time",
                               ])
    for k, row in targets_ms2.iterrows():
        seq = row["Sequence"]
        prec, frag = row["Precursor_Mz"], row["Product_Mz"]
        apext = results_ms1.loc[seq, "Apex_time"]
        start = results_ms1.loc[seq, f"Width_{argparser.width_2_pc}_pc_time_start"]
        stop = results_ms1.loc[seq, f"Width_{argparser.width_2_pc}_pc_time_end"]

        ms2_ext = exp.ms2.extract(prec)[start:stop]

        spec_apex_ms1 = ms2_ext[apext]
        tic_apext, tic_apexi = ms2_ext.tic.get_apex()


    p1 = exp.ms2.extract(prec)
    p1.tic.plot()

    pass


        

