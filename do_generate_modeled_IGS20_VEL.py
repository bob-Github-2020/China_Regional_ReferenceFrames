#!/usr/bin/python3
## 11-12-2025
# Final 10 RF reference frame: foreach f (XJYC XJHT XJYT XJQM XJRQ XJKE XJKC WUSH XJBC XJTZ)

import numpy as np

def calculate_global_velocities_from_helmert(station_lon, station_lat, dtx, dty, dtz, drx, dry, drz, R=6371.0):
    """
    Calculate IGS20 frame velocities at any geographic location using Helmert transformation parameters.
    """
    # Convert degrees to radians
    lon_rad = np.radians(station_lon)
    lat_rad = np.radians(station_lat)
    
    # 1. ROTATION COMPONENT
    omega_x = drx
    omega_y = dry  
    omega_z = drz
    
    # Create position vector of station (in km)
    r_x = R * np.cos(lat_rad) * np.cos(lon_rad)
    r_y = R * np.cos(lat_rad) * np.sin(lon_rad)
    r_z = R * np.sin(lat_rad)
    
    # Velocity from rotation: v_rot = Ω × r
    v_x_rot = omega_y * r_z - omega_z * r_y
    v_y_rot = omega_z * r_x - omega_x * r_z
    v_z_rot = omega_x * r_y - omega_y * r_x
    
    # 2. TRANSLATION COMPONENT - Reverse sign to recover IGS20 velocities
    v_x_trans = -dtx * 1e-3  # m/yr to km/yr, reverse sign
    v_y_trans = -dty * 1e-3
    v_z_trans = -dtz * 1e-3
    
    # 3. TOTAL VELOCITY in Cartesian coordinates
    v_x = v_x_rot + v_x_trans
    v_y = v_y_rot + v_y_trans
    v_z = v_z_rot + v_z_trans
    
    # 4. Convert to local topocentric coordinates
    V_e = -np.sin(lon_rad) * v_x + np.cos(lon_rad) * v_y
    V_n = -np.sin(lat_rad) * np.cos(lon_rad) * v_x - np.sin(lat_rad) * np.sin(lon_rad) * v_y + np.cos(lat_rad) * v_z
    V_u = np.cos(lat_rad) * np.cos(lon_rad) * v_x + np.cos(lat_rad) * np.sin(lon_rad) * v_y + np.sin(lat_rad) * v_z
    
    # Convert from km/yr to mm/yr
    return V_e * 1e6, V_n * 1e6, V_u * 1e6

def generate_site_name(lon, lat):
    """
    Generate site name in E80p1_N40p5 format
    """
    # Handle positive longitude (East)
    if lon >= 0:
        lon_str = f"E{abs(lon):.1f}"
    else:
        lon_str = f"W{abs(lon):.1f}"
    
    # Handle positive latitude (North)  
    if lat >= 0:
        lat_str = f"N{abs(lat):.1f}"
    else:
        lat_str = f"S{abs(lat):.1f}"
    
    # Replace decimal point with 'p'
    site_name = lon_str.replace('.', 'p') + '_' + lat_str.replace('.', 'p')
    return site_name

def main():
    # Tarim Basin transformation parameters
   
    dtx= 4.92541242193049E-04
    dty= 8.74936200868497E-03
    dtz= -1.06962235621836E-02
    drx= -1.69555619988227E-09
    dry= -1.17021809614531E-08
    drz= -3.6405596231366644E-09
    
    # Define grid boundaries for Tarim Basin
    lon_min, lon_max = 75.0, 96.0    # degrees
    lat_min, lat_max = 35.0, 43.0    # degrees
    grid_spacing = 1.0                # degrees
    
    # Create output file
    output_file = 'TRM_modeled_IGS20_Venu.vel'
    
    print(f"Generating velocity field for Tarim Basin...")
    print(f"Region: Longitude {lon_min}° to {lon_max}°, Latitude {lat_min}° to {lat_max}°")
    print(f"Grid spacing: {grid_spacing}° × {grid_spacing}°")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write('Long(deg) Lat(deg) Vel_E(mm/yr) Vel_N(mm/yr) Vel_U(mm/yr) 0 0 0 Site\n')
        
        # Generate grid points
        lons = np.arange(lon_min, lon_max + grid_spacing/2, grid_spacing)
        lats = np.arange(lat_min, lat_max + grid_spacing/2, grid_spacing)
        
        total_points = len(lons) * len(lats)
        point_count = 0
        
        for lat in lats:
            for lon in lons:
                # Calculate velocities
                V_e, V_n, V_u = calculate_global_velocities_from_helmert(
                    lon, lat, dtx, dty, dtz, drx, dry, drz
                )
                
                # Generate site name
                site_name = generate_site_name(lon, lat)
                
                # Write to file
                f.write(f'{lon:.4f} {lat:.4f} {V_e:.2f} {V_n:.2f} {V_u:.2f} 0.0 0.0 0.0 {site_name}\n')
                
                point_count += 1
                if point_count % 1000 == 0:
                    print(f"Processed {point_count}/{total_points} points...")
    
    print(f"\nVelocity field generation completed!")
    print(f"Total points: {point_count}")
    print(f"Output file: {output_file}")
    
    # Print some statistics
    print(f"\nGrid dimensions:")
    print(f"  Longitude points: {len(lons)}")
    print(f"  Latitude points: {len(lats)}")
    print(f"  Total grid cells: {total_points}")
    
    # Show first few points as example
    print(f"\nFirst 5 points as example:")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i in range(min(6, len(lines))):
            print(lines[i].strip())

if __name__ == "__main__":
    main()
