#!/usr/bin/python3
# Bob Wang, bob.g.wang@gmail.com

import numpy as np
import math
import glob
import os

def looks_like_year(token: str) -> bool:
    """
    Check if a token looks like a valid GPS year (e.g., 2010.8788, 2012.3121)
    """
    try:
        value = float(token)
        # GPS years are typically between 1990-2100 in decimal year format
        return 1990 <= value <= 2100
    except ValueError:
        return False

def xyz_to_llh(x, y, z, a=6378137.0, f=1/298.257223563):
    """
    Convert ECEF XYZ coordinates to longitude, latitude, and height (LLH)
    """
    # Ellipsoid parameters
    b = a * (1 - f)
    e2 = 1 - (b**2 / a**2)
    
    # Longitude
    lon = math.atan2(y, x)
    
    # Latitude and height (iterative method)
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1 - e2))
    
    for _ in range(10):
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(z, p * (1 - e2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-15:
            break
        lat = lat_new
    
    return math.degrees(lon), math.degrees(lat), h

def calculate_mean_llh_from_xyz(filename):
    """
    Calculate mean LLH from all XYZ coordinates in a file
    """
    x_coords = []
    y_coords = []
    z_coords = []
    skipped_lines = 0
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                parts = line.split()
                # Skip lines that don't have enough columns
                if len(parts) < 4:
                    skipped_lines += 1
                    continue
                
                # Skip lines that don't start with a valid GPS year
                first_token = parts[0]
                if not looks_like_year(first_token):
                    skipped_lines += 1
                    if skipped_lines <= 3:  # Only print first few skipped lines
                        print(f"    Skipping line {line_num+1}: {line.strip()[:60]}...")
                    continue
                
                # Try to parse the numeric data
                try:
                    x_coords.append(float(parts[1]))  # X coordinate
                    y_coords.append(float(parts[2]))  # Y coordinate  
                    z_coords.append(float(parts[3]))  # Z coordinate
                except (ValueError, IndexError):
                    skipped_lines += 1
                    continue
    
    if skipped_lines > 3:
        print(f"    ... and {skipped_lines - 3} more lines skipped")
    
    if not x_coords:
        raise ValueError(f"No valid numeric data found in {filename}")
    
    # Calculate mean coordinates
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords) 
    mean_z = np.mean(z_coords)
    
    # Convert to LLH
    mean_lon, mean_lat, mean_h = xyz_to_llh(mean_x, mean_y, mean_z)
    
    print(f"    Used {len(x_coords)} valid data points")
    
    return mean_lon, mean_lat, mean_h

def process_all_xyz_files():
    """
    Process all XYZ files and create Station.llh files
    """
    xyz_files = glob.glob("*.XYZ")
    
    if not xyz_files:
        print("No XYZ files found in current directory!")
        return
    
    print(f"Found {len(xyz_files)} XYZ files to process:")
    
    for xyz_file in xyz_files:
        print(f"Processing {xyz_file}...")
        
        # Get station name (first 4 characters of filename without extension)
        station_name = os.path.basename(xyz_file).replace('.XYZ', '')[:4]
        output_file = f"{station_name}.llh"
        
        try:
            # Calculate mean LLH
            lon, lat, h = calculate_mean_llh_from_xyz(xyz_file)
            
            # Write output file
            with open(output_file, 'w') as f:
                f.write(f"{lon:.9f} {lat:.9f} {h:.4f} {station_name}\n")
            
            print(f"  Station: {station_name}")
            print(f"  Mean position: Lon = {lon:.6f}°, Lat = {lat:.6f}°, H = {h:.3f}m")
            print(f"  Output: {output_file}")
            print()
            
        except Exception as e:
            print(f"  Error processing {xyz_file}: {e}")
            print()

# Alternative using pyproj (more accurate)
def calculate_mean_llh_pyproj(filename):
    """
    Calculate mean LLH using pyproj (more accurate)
    Requires: pip install pyproj
    """
    try:
        from pyproj import Transformer
        
        x_coords = []
        y_coords = [] 
        z_coords = []
        skipped_lines = 0
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    parts = line.split()
                    # Skip lines that don't have enough columns
                    if len(parts) < 4:
                        skipped_lines += 1
                        continue
                    
                    # Skip lines that don't start with a valid GPS year
                    first_token = parts[0]
                    if not looks_like_year(first_token):
                        skipped_lines += 1
                        continue
                    
                    # Try to parse the numeric data
                    try:
                        x_coords.append(float(parts[1]))
                        y_coords.append(float(parts[2]))
                        z_coords.append(float(parts[3]))
                    except (ValueError, IndexError):
                        skipped_lines += 1
                        continue
        
        if not x_coords:
            raise ValueError(f"No valid numeric data found in {filename}")
        
        # Calculate mean coordinates
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        mean_z = np.mean(z_coords)
        
        # Convert using pyproj
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")
        lon, lat, h = transformer.transform(mean_x, mean_y, mean_z)
        
        print(f"    Used {len(x_coords)} valid data points (pyproj)")
        
        return lon, lat, h
        
    except ImportError:
        print("pyproj not installed. Using built-in method.")
        return calculate_mean_llh_from_xyz(filename)

def process_all_xyz_files_pyproj():
    """
    Process all XYZ files using pyproj
    """
    xyz_files = glob.glob("*.XYZ")
    
    if not xyz_files:
        print("No XYZ files found in current directory!")
        return
    
    print(f"Found {len(xyz_files)} XYZ files to process (using pyproj):")
    
    for xyz_file in xyz_files:
        print(f"Processing {xyz_file}...")
        
        station_name = os.path.basename(xyz_file).replace('.XYZ', '')[:4]
        output_file = f"{station_name}.llh"
        
        try:
            lon, lat, h = calculate_mean_llh_pyproj(xyz_file)
            
            with open(output_file, 'w') as f:
                f.write(f"{lon:.9f} {lat:.9f} {h:.4f} {station_name}\n")
            
            print(f"  Station: {station_name}")
            print(f"  Mean position: Lon = {lon:.6f}°, Lat = {lat:.6f}°, H = {h:.3f}m")
            print(f"  Output: {output_file}")
            print()
            
        except Exception as e:
            print(f"  Error processing {xyz_file}: {e}")
            print()

if __name__ == "__main__":
    # Choose which method to use:
    
    # Method 1: Using built-in function (no dependencies)
    process_all_xyz_files()
    
    # Method 2: Using pyproj (more accurate - uncomment if installed)
    # process_all_xyz_files_pyproj()
