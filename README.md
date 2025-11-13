# PeakAnalyzer

Jupyter-based tools for analyzing experimental spectra and peaks.  
**PeakAnalyzer** helps you go from raw spectral data to cleaned, baseline-corrected, and fitted peaks with clear visualizations and exportable results.

Typical use cases include:
- Identifying peaks in noisy 1D spectra (e.g. IR, Raman, UV–Vis, MS traces)
- Performing baseline correction and smoothing
- Fitting peaks with simple models (e.g. Gaussian/Lorentzian)
- Comparing spectra across experiments

---

## Features

- 📊 **Interactive Jupyter workflow**  
  Work directly in notebooks and see each preprocessing and fitting step.

- 🧹 **Preprocessing utilities**  
  - Baseline correction  
  - Smoothing / denoising  
  - Normalization and cropping of spectral regions  

- 📈 **Peak analysis**  
  - Automatic peak detection in 1D spectra  
  - Manual refinement of peak positions  
  - Simple peak fitting (e.g. Gaussian, Lorentzian)  
  - Extraction of peak parameters (position, height, FWHM, area)

- 🖼️ **Visualization**  
  - Overlay raw and processed spectra  
  - Show detected peaks and fits  
  - Export figures for publications and reports

- 📁 **Simple I/O**  
  - Load spectra from common text formats (CSV / TXT)  
  - Save processed data and peak tables for further analysis

## Installation / Getting started

PeakAnalyzer is intentionally simple: it is just a Python file plus a Jupyter notebook.
You do not need a package installation or a complex setup.

To get started:

1. Make sure you have a working Python environment with Jupyter installed.
   Typical scientific installations (Anaconda, Miniconda, etc.) with numpy, scipy, pandas and matplotlib are sufficient.

2. Download or clone this repository so that you have:

   * the main Python file (for example: peakanalyzer.py)
   * the example Jupyter notebook (for example: PeakAnalyzer_example.ipynb)

3. Start Jupyter:

   * In a terminal, change into the folder with the notebook.
   * Run: jupyter lab
     or: jupyter notebook

4. Open the example notebook (for example: PeakAnalyzer_example.ipynb) in your browser.

5. Adjust the path to your own spectrum file (for example a CSV file with columns like "x" and "intensity") in the first cells of the notebook.

6. Run the notebook cells from top to bottom to:

   * load your spectrum,
   * preprocess it (baseline, smoothing, etc.),
   * find peaks,
   * optionally fit them,
   * and plot/save the results.

---

## Example use case: analyzing an IR spectrum from a CSV file

Imagine you have an IR spectrum exported from your instrument as a CSV file, called "sample_ir.csv", with two columns:

* x: wavenumber in cm⁻¹
* intensity: absorbance or transmittance

A typical workflow with PeakAnalyzer in the notebook is:

1. Load your data

   * Point the notebook to "sample_ir.csv".
   * The notebook reads the file and gives you a quick plot of the raw spectrum.

2. Clean up the spectrum

   * Apply a baseline correction to remove sloping or curved backgrounds.
   * Optionally smooth the signal (for example with a Savitzky–Golay filter) to reduce noise without destroying peak shapes.

3. Detect peaks

   * Use simple settings like minimum peak height, minimum distance between peaks and/or prominence.
   * The notebook highlights detected peaks directly on the spectrum, so you can visually check whether it makes chemical sense.

4. (Optional) Fit the peaks

   * For each detected peak, fit a simple model (for example Gaussian) to obtain peak position, height, area, and FWHM.
   * The results are summarized in a small table inside the notebook.

5. Export your results

   * Save a figure (for example "sample_ir_peaks.png") with the spectrum and marked peaks.
   * Save a table (for example "sample_ir_peaklist.csv") with the peak parameters, so you can use it later in your analysis, reports or publications.


If you later change how you like to preprocess or detect peaks, you can simply edit the notebook cells instead of touching any package configuration.


