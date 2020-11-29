# irtqc #
Performs calculation of parameters from LC-MS/MS run containing iRT peptides.

### Parameters ###
    -h, --help            show this help message and exit
    --mzml MZML           MzML file
    --targets TARGETS     Targets file
    --ms1-ppm MS1_PPM     MS1 extraction window in ppm
    --ms2-prec-tolerance  MS2 precursor tolerance
    --ms2-frag-tolerance  MS2 precursor tolerance
    --width-1-pc          Cromatographic width 1 in % of apex
    --width-2-pc          Cromatographic width 2 in % of apex
    --debug               Pickle cache input file, only for debug perposes
    
### Input ###
`targets` is a tab-separated table and contains a row for every target fragment. Columns are:
| Column name  | Description                                    |
|--------------|------------------------------------------------|
| Sequence     | Sequence of a peptide, used only for plotting  |
| Precursor_Mz | Peptide ion Mz                                 |
| Product_Mz   | Fragment ion Mz                                |

Other columns are ignored.

### Output ###
The tool generates 3 output files with names __MZML_Figs.pdf__, __MZML_MS1_table.csv__ and __MZML_MS2_table.csv__.

__MZML_Figs.csv__ contains a page for MS1 analysis plots with a row for each target and a page for MS2 analysis plots
 with a row for each target fragment. 

In each row in first page:
* First panel represents XIC for target extracted with _ms1-ppm_ window around target mass.
* Second panel is zoomed peaks and overlayed filtered/denoised peak. Calculated apex, and peakwidths at _width-1-pc_
% and _width-2-pc_% of apex height is marked. Also peak area at these levels of cut off is shown.
* At the third panel MS1 spectrum at apex is represented and target peak is marked.
* Fourth panel represents zoomed target peak with halfwidth and apex height are shown.
    
In each row in second page:
* First panel shows MS2 TIC for target precursor mass.
* Second represents zoomed peak for MS2 TIC with apex height shown.
* Third panel show MS2 spectrum at TIC apex with target fragments marked.
* Each of rightmost panels show zoomed one of target peaks with apex height and halfwidth marked.
    
__MZML_MS1_table.csv__ contains a row for every target peptide. Columns are:
| Column name                      | Description                                                                                                  |
|----------------------------------|--------------------------------------------------------------------------------------------------------------|
| Sequence                         | Sequence from `targets` input file                                                                           |
| Precursor_Mz                     | Peptide ion Mz from `targets` input file                                                                     |
| Apex_time                        | Time of XIC chromatigraphic peak apex in minutes                                                             |
| Width_`width-1-pc`_pc_time_start | Time when XIC chromatographic peak reaches `width-1-pc`% of its apex height                                  |
| Width_`width-1-pc`_pc_time_end   | Time when XIC chromatographic peak falls below `width-1-pc`% of its apex height                              |
| Width_`width-1-pc`_xic_area      | Area of XIC chromatographic peak between Width_`width-1-pc`_pc_time_start and Width_`width-1-pc`_pc_time_end |
| Width_`width-2-pc`_pc_time_start | Time when XIC chromatographic peak reaches `width-2-pc`% of its apex height                                  |
| Width_`width-2-pc`_pc_time_end   | Time when XIC chromatographic peak falls below `width-1-pc`% of its apex height                              |
| Width_`width-2-pc`_xic_area      | Area of XIC chromatographic peak between Width_`width-1-pc`_pc_time_start and Width_`width-1-pc`_pc_time_end |
| MS1_mass_apex_mz                 | Apex of target peak in MS1 spectrum at `Apex_time`                                                           |
| MS1_apex_height                  | Height of target peak in MS1 spectrum at `Apex_time`                                                         |
| MS1_peak_halfwidth               | Halfwidht of target peak in MS1 spectrum at `Apex_time`                                                      |
| MS1_peak_area                    | Area of target peak in MS1 spectrum at `Apex_time`                                                           |
| TIC_MS2                          | MS2 TIC current for target peak for MS2 spectrum nearest to `Apex_time`                                      |


__MZML_MS2_table.csv__ contains a row for every target peptide. Columns are:
| Column name        | Description                                                              |
|--------------------|--------------------------------------------------------------------------|
| Sequence           | Sequence from  `targets` input file                                      |
| Precursor_Mz       | Peptide ion Mz from  `targets` input file                                |
| Product_Mz         | Fragment ion Mz from  `targets` input file                               |
| MS2_TIC_Apex_time  | MS2 TIC chromatigraphic peak apex in minutes                             |
| MS2_mass_apex_mz   | Apex of fragment peak in MS2 spectrum at  `Apex_time`                    |
| MS2_apex_height    | Height of fragment peak in MS2 spectrum at  `Apex_time`                  |
| MS2_peak_halfwidth | Height of fragment peak in MS2 spectrum at `Apex_time`                   |
| MS2_peak_area      | Area of fragment peak in MS2 spectrum at `Apex_time`                     |
    
    
