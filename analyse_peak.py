import pandas as pd
from os import path
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
import glob


def process_files():
    # set current directory and directory to resources
    current_directory = os.getcwd()
    resource_path = os.path.join(current_directory, "resources")
    figure_path = os.path.join(current_directory, "figures")
    files_path = os.path.join(current_directory, "files")

    if not os.path.exists(resource_path):
        os.makedirs(resource_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    if not os.path.exists(files_path):
        os.makedirs(files_path)

    files = [
        f for f in os.listdir(resource_path) if path.isfile(path.join(resource_path, f))
    ]
    dataset = {}
    for file in files:
        file_path = os.path.join(resource_path, file)
        try:
            # Read the CSV file
            data = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=13)

            # If the data has more than one column, assign column names
            if data.shape[1] > 1:
                data.columns = ["wavelength", "intensity"]
                dataset[file] = data
            else:
                print(f"Error: Only one column found in file {file}")
        except Exception as e:
            print(f"Failed to process {file}: {str(e)}")

    # sort dataset by filename
    dataset = dict(sorted(dataset.items()))
    return dataset, figure_path, files_path


def lorentzian_with_offset(x, x0, a, gamma, y0):
    return y0 + (2 * a * np.pi) * (gamma**2 / ((x - x0) ** 2 + gamma**2))


def create_figure_of_fit(data, x_data, y_data, popt, file, figure_path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        data["wavelength"][(data["wavelength"] > 500) & (data["wavelength"] < 900)],
        data["intensity"][(data["wavelength"] > 500) & (data["wavelength"] < 900)],
        "b.",
        label="data",
    )  # Original data
    plt.plot(x_data, y_data, "g.", label="fitting range")  # Fitting range
    # Plot the fit
    x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
    y_fit = lorentzian_with_offset(x_fit, *popt)
    plt.plot(
        x_fit,
        y_fit,
        "r-",
        label=r"Fit: $y = \frac{{%.2f \cdot \pi \cdot %.2f^2}}{{(x - %.2f)^2 + %.2f^2}} + %.5f$"
        % (popt[1], popt[2], popt[0], popt[2], popt[3]),
    )
    plt.xlabel("wavelength [nm]")
    plt.ylabel("intensity [a.u.]")
    plt.title(f"Lorentzian Fit with Offset for {file}")
    plt.legend(loc="upper right", frameon=False)
    plt.savefig(f"{figure_path}/{file}_fit.png", dpi=300)
    plt.close()


def average_value(range, data, column):
    return data.loc[range, column].mean()


def fit_lorentzian(
    dataset,
    figure_path,
    files_path,
    fit_from=400,
    fit_to=950,
    steps_from_peak=20,
    create_figure=True,
    create_video=False,
    number_of_points_for_peak_average=2,
    number_of_points_for_absorbance_average=2,
):
    file_number = 0
    number_of_files = len(dataset)
    print(f"Found {number_of_files} files in the dataset")
    for file, data in dataset.items():
        print(f"Data for file {file} loaded")
        filtered_data = data[
            (data["wavelength"] > fit_from) & (data["wavelength"] < fit_to)
        ]
        peak_index = filtered_data["intensity"].idxmax()
        peak_position = filtered_data.loc[peak_index, "wavelength"]
        filtered_data = filtered_data[
            (filtered_data["wavelength"] > (peak_position - steps_from_peak))
        ]
        x_data = filtered_data["wavelength"]
        y_data = filtered_data["intensity"]
        x0_initial = peak_position
        a_initial = y_data.max()
        gamma_initial = (
            x_data.max() - x_data.min()
        ) / 20  # Rough estimate for gamma (HWHM)
        y0_initial = y_data.min()  # Initial estimate for y-offset
        bounds = (
            [np.min(x_data), 0, 0, -np.inf],
            [np.max(x_data), np.inf, np.inf, np.inf],
        )
        popt, pcov = curve_fit(
            lorentzian_with_offset,
            x_data,
            y_data,
            p0=[x0_initial, a_initial, gamma_initial, y0_initial],
            bounds=bounds,
        )

        residuals = y_data - lorentzian_with_offset(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        file_number += 1
        print(f"Analyzing file {file_number} of {number_of_files}")
        # calculate the average peak position and absorbance from the raw data take + number_of_points_for_peak_average/2 and - number_of_points_for_peak_average/2
        if (
            number_of_points_for_peak_average is not None
            and number_of_points_for_peak_average != 1
        ):
            peak_range = range(
                peak_index - number_of_points_for_peak_average // 2,
                peak_index + number_of_points_for_peak_average // 2,
            )
            average_raw_peak_position = average_value(peak_range, data, "wavelength")
            average_raw_peak_intensity = average_value(peak_range, data, "intensity")
        else:
            average_raw_peak_position = "N/A"
            average_raw_peak_intensity = "N/A"
        if (
            number_of_points_for_absorbance_average is not None
            and number_of_points_for_absorbance_average != 1
        ):
            absorbance_range = range(
                400 - number_of_points_for_absorbance_average // 2,
                400 + number_of_points_for_absorbance_average // 2,
            )
            average_raw_absorbance = average_value(absorbance_range, data, "intensity")
        else:
            average_raw_absorbance = "N/A"

        peak_intensity = lorentzian_with_offset(popt[0], *popt)
        if create_figure:
            create_figure_of_fit(data, x_data, y_data, popt, file, figure_path)
        if not os.path.exists(f"{files_path}/results.csv"):
            with open(f"{files_path}/results.csv", "w") as f:
                f.write(
                    "Filename    Peakposition    Peakintensity    FWHM    Peakposition_gemittelt    Peakintensity_gemittelt    Absorbanz_400nm_gemittelt    R^2_of_fit\n"
                )
        with open(f"{files_path}/results.csv", "a") as f:
            f.write(
                f"{file}    {popt[0]}  {peak_intensity} {popt[2]}   {average_raw_peak_position}    {average_raw_peak_intensity}    {average_raw_absorbance}    {r_squared}\n"
            )
    if create_video:
        create_video(figure_path, "output_video.mp4", fps=10)
    print("All files processed")


def create_video(folder_path, output_path, fps=30):
    images = sorted(glob.glob(os.path.join(folder_path, "*.png")), key=os.path.getmtime)
    if not images:
        print("No images found. Check the folder path and file extensions.")
        return
    frame = cv2.imread(images[0])
    if frame is None:
        print(
            "Failed to read the first image. Ensure the file is accessible and not corrupted."
        )
        return
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in images:
        video_frame = cv2.imread(image)
        if video_frame is not None:
            video.write(video_frame)
        else:
            print(f"Warning: Failed to read image {image}")
    video.release()
    print(f"Video created successfully and saved to {output_path}")


dataset, figure_path, files_path = process_files()

fit_lorentzian(
    dataset=dataset,
    figure_path=figure_path,
    files_path=files_path,
    create_figure=True,
    create_video=True,
    fit_from=400,
    fit_to=950,
    steps_from_peak=20,
    number_of_points_for_peak_average=10,
    number_of_points_for_absorbance_average=10,
)
