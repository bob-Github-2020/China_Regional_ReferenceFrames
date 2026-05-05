#!/usr/bin/python3
## Bob Wang, bob.g.wang@gmail.com

## 10-10-2025, China20RF--CHN20
## Read in 7 columns or 4 columns NEU files

## Transform NEU time series (TENU) from the IGS references (global) to local reference frame

## The input includes the (1) Long, Lat. of the station; (2) 7 parameters, (3) and the IGS20 ENU time series
## I did comparisons with the TXAG ENU time series from IGS14 to GOM20, it works perfectly.

import math
import numpy as np
import pandas as pd
import os
import json

def load_reference_frames(param_file="reference_frames.json"):
    """
    Load reference frame parameters from a JSON file
    Format of JSON file:
    {
        "CHN20": {
            "t0": 2020.0,
            "dtx": 7.6537010355933334E-04,
            "dty": 8.5818487895197339E-04,
            "dtz": -1.3855495394016564E-03,
            "drx": -6.4669113756026333E-10,
            "dry": -2.7972438316742459E-09,
            "drz": 4.2044610770199437E-09,
            "drs": 0.0000000000000000
        },
        "NWC20": {
            "t0": 2020.0,
            "dtx": -3.1969566939800249E-03,
            "dty": -2.3148730062595318E-03,
            "dtz": 1.4920654200029568E-03,
            "drx": 8.5139120141373714E-10,
            "dry": -5.9730873980248107E-09,
            "drz": 1.9037573188301016E-09,
            "drs": 0.0000000000000000
        }
    }
    """
    try:
        with open(param_file, 'r') as f:
            reference_frames = json.load(f)
        print(f"Loaded {len(reference_frames)} reference frames from {param_file}")
        return reference_frames
    except FileNotFoundError:
        print(f"Error: Parameter file {param_file} not found!")
        print("Please create a JSON file with reference frame parameters.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {param_file}!")
        return {}
## Creat reference_frames.json
## rm reference_frames.json, if made any changes here
def create_sample_parameter_file(param_file="reference_frames.json"):
    """Create a sample parameter file if it doesn't exist"""
    sample_frames = {
        # 10-30-2025, 137 RF
        "CHN20": {
            "t0": 2020.0,
            # "dtx": 1.1543004132820701E-03,
            # "dty": 8.7255758171650899E-04,
            # "dtz": -1.1424036188835301E-03,
            # "drx": -7.8085600959880131E-10,
            # "dry": -2.5205209486784212E-09,
            # "drz": 4.3325671477519133E-09,
            "dtx": 1.1374256718928370E-03,
            "dty": 9.0981504265975259E-04,
            "dtz": -1.1864748995835017E-03,
            "drx": -7.9477524536606623E-10,
            "dry": -2.5018560411572973E-09,
            "drz": 4.3583072157488161E-09,
            "drs": 0.0000000000000000
        },
        "TSN20": {
            "t0": 2020.0,
            "dtx": -2.7844586745028833E-03,
            "dty": -3.0008528045700967E-02,
            "dtz": 3.0973495549614628E-02,
            "drx": 7.3126816307216767E-09,
            "dry": -4.9271074998137855E-09,
            "drz": 2.7924021468633099E-09,
            "drs": 0.0000000000000000
        },
        "TRM20": {
            "t0": 2020.0,
            "dtx": 4.9254124219304946E-04,
            "dty": 8.7493620086849730E-03,
            "dtz": -1.0696223562183579E-02,
            "drx": -1.6955561998822696E-09,
            "dry": -1.1702180961453092E-08,
            "drz": -3.6405596231366644E-09,
            "drs": 0.0000000000000000
        },
        "ALS20": {
            "t0": 2020.0,
            "dtx": -7.3957899872959063E-03,
            "dty": -1.0440364449249945E-02,
            "dtz": 9.5719161668034458E-03,
            "drx": 1.9484242087310248E-09,
            "dry": -3.3345635315773299E-09,
            "drz": 5.3490587855555357E-09,
            "drs": 0.0000000000000000
        },
        "JGR20": {
            "t0": 2020.0,
            "dtx": -1.1112884598345018E-03,
            "dty": -5.2555896745076960E-03,
            "dtz": 4.6693113838390367E-03,
            "drx": 1.5247362221932615E-09,
            "dry": -2.5482999470750451E-10,
            "drz": 6.8206042550428445E-09,
            "drs": 0.0000000000000000
        },
        "SCS20": {
            "t0": 2020.0,
             "dtx": -7.5271870373108379E-03,
             "dty": -9.5022784648809923E-04,
             "dtz": -5.6359208139182531E-03,
             "drx": -1.2811954988340727E-09,
             "dry": -3.6983412054869651E-09,
             "drz": 5.4713417063949494E-09,
             "drs": 0.0000000000000000
        },  
        "NEC20": {
            "t0": 2020.0,
            "dtx": -2.3160550224010151E-03,
            "dty": -2.7532763958112686E-03,
            "dtz": 3.9756053027034308E-04,
            "drx": -7.3337782918385403E-10,
            "dry": -1.9295840527471817E-09,
            "drz": 5.5712171180111659E-09,
            "drs": 0.0000000000000000
        }, 
        ## NCH20, 40RF
        "NCH20": {
            "t0": 2020.0,
            "dtx": -8.0155625786492764E-03,
            "dty": -2.6289532032148411E-03,
            "dtz": -3.1380200570955579E-03,
            "drx": -1.2249769583231260E-09,
            "dry": -2.5824575872777383E-09,
            "drz": 6.4892993380732547E-09,
            "drs": 0.0000000000000000
        }, 
        "SCH20": {
            "t0": 2020.0,
            "dtx": -1.3463810363962771E-03,
            "dty": -5.3660721387459027E-04,
            "dtz": -1.1301089050630500E-03,
            "drx": -8.2939733743576191E-10,
            "dry": -2.9101371648476450E-09,
            "drz": 4.9601468488461615E-09,
            "drs": 0.0000000000000000
        },              
        "SWC20": {
            "t0": 2020.0,
            "dtx": 2.0531686633038327E-02,
            "dty": 1.6278807542576580E-03,
            "dtz": -6.2365342032911857E-04,
            "drx": 0.0,
            "dry": 0.0,
            "drz": 0.0,
            "drs": 0.0000000000000000
        }
    }

 
    with open(param_file, 'w') as f:
        json.dump(sample_frames, f, indent=4)
    print(f"Created sample parameter file: {param_file}")
    return sample_frames

# Function to convert ENU time series from IGS to local reference frame
def enu_IGS_to_Local(input_file, output_file, lat, lon, t0, dtx, dty, dtz, drx, dry, drz, drs):
    # Read input file and detect if it has uncertainties
    try:
        # First, detect the number of columns in the file
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        # Count columns in second line (data line)
        second_line_cols = len(second_line.split())
        has_uncertainties = (second_line_cols >= 7)  # At least 7 columns means uncertainties
        
        # Reset file pointer and read with appropriate column names
        if has_uncertainties:
            df = pd.read_csv(input_file, delim_whitespace=True, header=0, 
                           names=["Decimal_Year", "NS", "EW", "UD", "Sigma_NS", "Sigma_EW", "Sigma_UD"])
        else:
            # For 4-column files, create Sigma columns with zeros
            df = pd.read_csv(input_file, delim_whitespace=True, header=0,
                           names=["Decimal_Year", "NS", "EW", "UD"])
            df["Sigma_NS"] = 0.0
            df["Sigma_EW"] = 0.0
            df["Sigma_UD"] = 0.0
            
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Ensure the DataFrame is sorted by Decimal_Year
    df = df.sort_values(by="Decimal_Year").reset_index(drop=True)
    
    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2

    # Calculate the radius of curvature in the prime vertical
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)

    # Calculate the reference point ECEF coordinates (assuming ellipsoidal height h = 0)
    X0 = (N) * math.cos(lat_rad) * math.cos(lon_rad)
    Y0 = (N) * math.cos(lat_rad) * math.sin(lon_rad)
    Z0 = (N * (1 - e2)) * math.sin(lat_rad)

    # Corrected rotation matrix from ENU to ECEF
    R_enu = np.array([
        [-math.sin(lon_rad), -math.sin(lat_rad) * math.cos(lon_rad), math.cos(lat_rad) * math.cos(lon_rad)],
        [math.cos(lon_rad), -math.sin(lat_rad) * math.sin(lon_rad), math.cos(lat_rad) * math.sin(lon_rad)],
        [0, math.cos(lat_rad), math.sin(lat_rad)]
    ])

    # Prepare lists to hold transformed ENU coordinates
    Local_ns = []
    Local_ew = []
    Local_ud = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract time and ENU coordinates in IGS14
        t = row["Decimal_Year"]
        E = row["EW"] / 100.0  # Convert from cm to m
        N = row["NS"] / 100.0  # Convert from cm to m
        U = row["UD"] / 100.0  # Convert from cm to m
        
        # Convert ENU to ECEF displacement in IGS14
        enu_vector = np.array([E, N, U])
        ecef_displacement_IGS = R_enu.dot(enu_vector)

        # Get the full ECEF coordinates by adding the reference point
        X = X0 + ecef_displacement_IGS[0]
        Y = Y0 + ecef_displacement_IGS[1]
        Z = Z0 + ecef_displacement_IGS[2]

        # Apply the rate-based transformation to GOM20 (translation + rotation)
        delta_t = t - t0

        # Translation component
        dX = dtx * delta_t
        dY = dty * delta_t
        dZ = dtz * delta_t

        # Rotation component
        dX_rot = drz * delta_t * Y - dry * delta_t * Z
        dY_rot = -drz * delta_t * X + drx * delta_t * Z
        dZ_rot = dry * delta_t * X - drx * delta_t * Y

        # Scale component (drs is typically 0, so no effect here)
        # dX_scale = drs * X * delta_t  # Not implemented as drs=0
        # dY_scale = drs * Y * delta_t
        # dZ_scale = drs * Z * delta_t

        # Total change in ECEF coordinates
        X_Local = X + dX + dX_rot
        Y_Local = Y + dY + dY_rot
        Z_Local = Z + dZ + dZ_rot

        # Convert transformed ECEF displacement back to ENU (relative to original point)
        ecef_displacement_Local = np.array([X_Local - X0, Y_Local - Y0, Z_Local - Z0])
        enu_displacement_Local = R_enu.T.dot(ecef_displacement_Local)

        # Append the transformed ENU coordinates to the respective lists (convert back to cm)
        Local_ew.append(enu_displacement_Local[0] * 100.0)
        Local_ns.append(enu_displacement_Local[1] * 100.0)
        Local_ud.append(enu_displacement_Local[2] * 100.0)

    # Add the transformed ENU coordinates to the DataFrame
    df["NS_Local"] = Local_ns
    df["EW_Local"] = Local_ew
    df["UD_Local"] = Local_ud

    # Standardize the transformed time series by removing the mean of the first 30 days
    first_30_days = df[df['Decimal_Year'] <= df['Decimal_Year'][0] + (30 / 365.25)]
    ns_mean = first_30_days['NS_Local'].mean()
    ew_mean = first_30_days['EW_Local'].mean()
    ud_mean = first_30_days['UD_Local'].mean()

    df['NS_Local'] = df['NS_Local'] - ns_mean
    df['EW_Local'] = df['EW_Local'] - ew_mean
    df['UD_Local'] = df['UD_Local'] - ud_mean

    # Write the output DataFrame to a new file with the same format as the input
    df = df[["Decimal_Year", "NS_Local", "EW_Local", "UD_Local", "Sigma_NS", "Sigma_EW", "Sigma_UD"]]
    df.columns = ["Decimal_Year", "NS", "EW", "UD", "Sigma_NS", "Sigma_EW", "Sigma_UD"]
    df.to_csv(output_file, index=False, sep=' ', float_format='%.4f')
    print(f"Transformed ENU time series saved to '{output_file}'")

## read the LLH file, long, lat, H, Station    
def load_station_coords(llh_file, station):
    """
    Load station coordinates from combined LLH file
    Format: Longitude(deg) Latitude(deg) Height(m) StationName
    """
    try:
        with open(llh_file, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2.5:      ## minimum 2.5 year data
                    # Check if station name matches (last column)
                    if parts[-1] == station:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        # height = float(parts[2])  # Not needed for ENU transformation
                        return lat, lon
            raise ValueError(f"Station {station} not found in {llh_file}")
    except FileNotFoundError:
        raise ValueError(f"LLH file {llh_file} not found")


# Main function to process multiple files
def main(llh_file="China_GPS_llh.LLH", local_frame="CHN20", param_file="reference_frames.json"):
    # Load reference frame parameters
    reference_frames = load_reference_frames(param_file)
    
    # If parameter file doesn't exist, create a sample one
    if not reference_frames:
        reference_frames = create_sample_parameter_file(param_file)
        if not reference_frames:
            print("No reference frames available. Exiting.")
            return
    
    # Check if requested local frame exists
    if local_frame not in reference_frames:
        print(f"Error: Reference frame '{local_frame}' not found in {param_file}")
        print(f"Available frames: {', '.join(reference_frames.keys())}")
        return
    
    # Get transformation parameters for the selected local frame
    params = reference_frames[local_frame]
    print(f"Using reference frame: {local_frame}")

    # Get all .col files in the current directory
    input_files = [f for f in os.listdir('.') if f.endswith('_cm.col')]
    
    if not input_files:
        print("No *_cm.col files found in current directory")
        return

    for input_file in input_files:
        print(f"Processing: {input_file}")
        # Extract station name from filename (assuming format like "OKCB_IGS14_neu_cm.col")
        station = input_file.split('_')[0]
        
        # Load station coordinates
        try:
            lat, lon = load_station_coords(llh_file, station)
        except ValueError as e:
            print(e)
            continue

        # Set input and output filenames, replace "IGS14" with local_frame
        output_file = input_file.replace("IGS20", local_frame)

        # Transform the time series
        enu_IGS_to_Local(input_file, output_file, lat, lon, **params)

if __name__ == "__main__":
    # You can now easily change the reference frame here
     main(llh_file="China_GPS_llh.LLH", local_frame="CHN20")
    #main(llh_file="Taiwan_GPS_llh.LLH", local_frame="CHN20")
    
    # Or call with different frames:
    # main(local_frame="CHN20")
    # main(local_frame="NWC20") 
    # main(local_frame="TLM20")
