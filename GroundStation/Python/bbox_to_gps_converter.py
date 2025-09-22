#!/usr/bin/env python3
"""
Bounding Box to GPS Coordinate Converter

This script converts bounding box coordinates in the image plane to GPS coordinates
using vehicle attitude, gimbal attitude, and vehicle GPS position as reference.

Required libraries (install with pip):
- numpy
- scipy
- pymap3d

Install command: pip install numpy scipy pymap3d
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import pymap3d
import math
from typing import Dict, Tuple, Optional, List


class BBoxToGPSConverter:
    """
    Converts bounding box coordinates to GPS coordinates using camera geometry
    and vehicle/gimbal orientation data.
    """
    
    def __init__(self, camera_params: Dict):
        """
        Initialize the converter with camera parameters.
        
        Args:
            camera_params: Dictionary containing camera calibration parameters
                - 'fx': Focal length in pixels (x-direction)
                - 'fy': Focal length in pixels (y-direction) 
                - 'cx': Principal point x-coordinate (pixels)
                - 'cy': Principal point y-coordinate (pixels)
                - 'image_width': Image width in pixels
                - 'image_height': Image height in pixels
        """
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params['cx']
        self.cy = camera_params['cy']
        self.image_width = camera_params['image_width']
        self.image_height = camera_params['image_height']
        
        # Validation parameters
        self.max_ground_distance = 2000  # Maximum reasonable distance in meters
        self.min_altitude = 3           # Minimum vehicle altitude in meters
        self.max_altitude = 500         # Maximum reasonable altitude in meters
        
    def bbox_center_to_gps(self, 
                          bbox: Tuple[int, int, int, int],
                          vehicle_attitude: Dict[str, float],
                          gimbal_attitude: Dict[str, float],
                          vehicle_gps: Dict[str, float],
                          use_bbox_center: bool = True) -> Optional[Dict]:
        """
        Convert bounding box coordinates to GPS coordinates.
        
        Args:
            bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
            vehicle_attitude: Dict with keys 'roll', 'pitch', 'yaw' in degrees
            gimbal_attitude: Dict with keys 'roll', 'pitch', 'yaw' in degrees  
            vehicle_gps: Dict with keys 'latitude', 'longitude', 'altitude'
            use_bbox_center: If True, use center of bbox. If False, use bottom center
            
        Returns:
            Dictionary with GPS coordinates and metadata, or None if calculation fails
        """
        
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox
        
        # Validate bounding box
        if not self._validate_bbox(bbox):
            print("Error: Invalid bounding box coordinates")
            return None
            
        # Calculate target pixel coordinates
        if use_bbox_center:
            # Use center of bounding box
            x_pixel = (x_min + x_max) / 2.0
            y_pixel = (y_min + y_max) / 2.0
        else:
            # Use bottom center of bounding box (often better for ground objects)
            x_pixel = (x_min + x_max) / 2.0
            y_pixel = y_max
            
        # Convert to GPS coordinates
        result = self.pixel_to_gps(
            x_pixel, y_pixel,
            vehicle_attitude, gimbal_attitude, vehicle_gps
        )
        
        if result:
            # Add bounding box information
            result['bbox'] = bbox
            result['bbox_center_pixel'] = (x_pixel, y_pixel)
            result['bbox_width'] = x_max - x_min
            result['bbox_height'] = y_max - y_min
            
        return result
        
    def pixel_to_gps(self,
                    x_pixel: float, y_pixel: float,
                    vehicle_attitude: Dict[str, float],
                    gimbal_attitude: Dict[str, float], 
                    vehicle_gps: Dict[str, float]) -> Optional[Dict]:
        """
        Convert pixel coordinates to GPS coordinates.
        
        Args:
            x_pixel, y_pixel: Pixel coordinates in image
            vehicle_attitude: Dict with keys 'roll', 'pitch', 'yaw' in degrees
            gimbal_attitude: Dict with keys 'roll', 'pitch', 'yaw' in degrees
            vehicle_gps: Dict with keys 'latitude', 'longitude', 'altitude'
            
        Returns:
            Dictionary with GPS coordinates and metadata, or None if calculation fails
        """
        
        # Input validation
        if not self._validate_inputs(vehicle_attitude, gimbal_attitude, vehicle_gps):
            return None
            
        try:
            # Extract values
            lat_vehicle = vehicle_gps['latitude']
            lon_vehicle = vehicle_gps['longitude'] 
            alt_vehicle = vehicle_gps['altitude']
            
            # Create camera ray in camera coordinate system
            # Camera frame: X=right, Y=down, Z=forward (optical axis)
            x_cam = (x_pixel - self.cx) / self.fx
            y_cam = (y_pixel - self.cy) / self.fy
            z_cam = 1.0  # Normalized focal length
            
            # Camera ray vector (normalized)
            ray_camera = np.array([[x_cam], [y_cam], [z_cam]])
            ray_camera = ray_camera / np.linalg.norm(ray_camera)
            
            # Create rotation matrices
            # Vehicle attitude: rotation from vehicle body frame to world frame (NED)
            vehicle_rotation = R.from_euler('ZYX', [
                vehicle_attitude['yaw'],
                vehicle_attitude['pitch'],
                vehicle_attitude['roll']
            ], degrees=True)
            
            # Gimbal attitude: rotation from camera frame to vehicle body frame
            gimbal_rotation = R.from_euler('ZYX', [
                gimbal_attitude['yaw'],
                gimbal_attitude['pitch'], 
                gimbal_attitude['roll']
            ], degrees=True)
            
            # Combined rotation: camera frame to world frame
            combined_rotation = vehicle_rotation * gimbal_rotation
            C_camera_to_world = combined_rotation.as_matrix()
            
            # Transform ray to world coordinate system (NED)
            ray_world = np.matmul(C_camera_to_world, ray_camera)
            
            # Check if ray points downward (positive Z in NED frame)
            if ray_world[2, 0] <= 0:
                print("Warning: Ray points upward, no ground intersection possible")
                return None
                
            # Calculate intersection with ground plane
            # Assuming ground at elevation 0 (can be modified for terrain)
            # Ray equation: P = P0 + t * direction
            # Ground plane: Z = 0 (NED frame)
            t = alt_vehicle / ray_world[2, 0]
            
            # Ground intersection point in NED frame relative to vehicle
            ground_point_ned = t * ray_world
            
            # Validate reasonable distance
            horizontal_distance = math.sqrt(
                ground_point_ned[0, 0]**2 + ground_point_ned[1, 0]**2
            )
            
            if horizontal_distance > self.max_ground_distance:
                print(f"Warning: Calculated distance {horizontal_distance:.1f}m exceeds maximum")
                return None
                
            # Convert NED coordinates to GPS
            target_lat, target_lon, target_alt = pymap3d.ned2geodetic(
                ground_point_ned[0, 0],  # North
                ground_point_ned[1, 0],  # East  
                -alt_vehicle,            # Down (negative altitude)
                lat_vehicle, lon_vehicle, alt_vehicle
            )
            
            # Calculate additional metrics
            bearing = math.atan2(ground_point_ned[1, 0], ground_point_ned[0, 0])
            bearing_degrees = math.degrees(bearing)
            if bearing_degrees < 0:
                bearing_degrees += 360
                
            elevation_angle = math.degrees(math.atan2(-ray_world[2, 0], 
                                                     math.sqrt(ray_world[0, 0]**2 + ray_world[1, 0]**2)))
            
            # Estimate uncertainty
            uncertainty = self._estimate_uncertainty(
                alt_vehicle, horizontal_distance, elevation_angle
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                alt_vehicle, elevation_angle, horizontal_distance
            )
            
            # Compile results
            result = {
                'target_latitude': target_lat,
                'target_longitude': target_lon,
                'target_altitude': 0.0,  # Assumed ground level
                'horizontal_distance_m': horizontal_distance,
                'bearing_degrees': bearing_degrees,
                'elevation_angle_degrees': elevation_angle,
                'uncertainty_m': uncertainty,
                'confidence_score': confidence,
                'vehicle_position': {
                    'latitude': lat_vehicle,
                    'longitude': lon_vehicle,
                    'altitude': alt_vehicle
                },
                'pixel_coordinates': {
                    'x': x_pixel,
                    'y': y_pixel
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in GPS conversion: {e}")
            return None
            
    def _validate_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Validate bounding box coordinates."""
        x_min, y_min, x_max, y_max = bbox
        
        # Check bounds
        if (x_min < 0 or y_min < 0 or 
            x_max >= self.image_width or y_max >= self.image_height):
            return False
            
        # Check ordering
        if x_min >= x_max or y_min >= y_max:
            return False
            
        # Check minimum size
        if (x_max - x_min) < 5 or (y_max - y_min) < 5:
            return False
            
        return True
        
    def _validate_inputs(self, vehicle_attitude: Dict, gimbal_attitude: Dict, 
                        vehicle_gps: Dict) -> bool:
        """Validate input parameters."""
        
        # Check required keys
        required_attitude_keys = ['roll', 'pitch', 'yaw']
        required_gps_keys = ['latitude', 'longitude', 'altitude']
        
        for key in required_attitude_keys:
            if key not in vehicle_attitude or key not in gimbal_attitude:
                print(f"Error: Missing attitude key: {key}")
                return False
                
        for key in required_gps_keys:
            if key not in vehicle_gps:
                print(f"Error: Missing GPS key: {key}")
                return False
                
        # Validate ranges
        alt = vehicle_gps['altitude']
        if alt < self.min_altitude or alt > self.max_altitude:
            print(f"Warning: Vehicle altitude {alt}m outside reasonable range")
            
        # Validate GPS coordinates
        lat = vehicle_gps['latitude']
        lon = vehicle_gps['longitude']
        if abs(lat) > 90 or abs(lon) > 180:
            print("Error: Invalid GPS coordinates")
            return False
            
        return True
        
    def _estimate_uncertainty(self, altitude: float, horizontal_distance: float, 
                            elevation_angle: float) -> float:
        """Estimate position uncertainty in meters."""
        
        # Uncertainty sources (simplified model)
        gps_uncertainty = 3.0      # Vehicle GPS uncertainty (meters)
        attitude_uncertainty = 0.5  # Attitude uncertainty (degrees)
        pixel_uncertainty = 1.0    # Pixel detection uncertainty
        
        # Convert to radians
        attitude_error_rad = math.radians(attitude_uncertainty)
        pixel_error_rad = pixel_uncertainty / self.fx  # Simplified
        
        # Angular uncertainty contribution
        angular_contribution = horizontal_distance * attitude_error_rad
        
        # Pixel uncertainty contribution  
        pixel_contribution = altitude * pixel_error_rad / math.sin(math.radians(abs(elevation_angle)))
        
        # Geometric dilution factor
        geometric_factor = 1.0 / math.sin(math.radians(abs(elevation_angle)))
        
        # Combine uncertainties
        total_uncertainty = math.sqrt(
            (angular_contribution * geometric_factor)**2 +
            pixel_contribution**2 +
            gps_uncertainty**2
        )
        
        return total_uncertainty
        
    def _calculate_confidence(self, altitude: float, elevation_angle: float, 
                            horizontal_distance: float) -> float:
        """Calculate confidence score (0-1)."""
        
        # Higher confidence for:
        # - Higher altitude (better angular resolution)
        # - Steeper elevation angle (less geometric uncertainty)
        # - Shorter distances (less error propagation)
        
        altitude_factor = min(altitude / 100.0, 1.0)
        angle_factor = min(abs(elevation_angle) / 45.0, 1.0)
        distance_factor = max(0, 1.0 - horizontal_distance / 1000.0)
        
        confidence = (altitude_factor + angle_factor + distance_factor) / 3.0
        return max(0.1, min(1.0, confidence))


def example_usage():
    """Example usage of the BBoxToGPSConverter."""
    
    # Camera parameters for DJI thermal camera (example values)
    camera_params = {
        'fx': 773.89,  # Focal length in pixels (x-direction)
        'fy': 773.89,  # Focal length in pixels (y-direction)
        'cx': 320.0,   # Principal point x-coordinate
        'cy': 256.0,   # Principal point y-coordinate
        'image_width': 640,   # Image width
        'image_height': 512   # Image height
    }
    
    # Initialize converter
    converter = BBoxToGPSConverter(camera_params)
    
    # Example detection bounding box (x_min, y_min, x_max, y_max)
    fire_detection_bbox = (280, 200, 360, 280)
    
    # Vehicle attitude (degrees)
    vehicle_attitude = {
        'roll': 2.1,
        'pitch': -5.3,
        'yaw': 45.7
    }
    
    # Gimbal attitude (degrees) 
    gimbal_attitude = {
        'roll': 0.2,
        'pitch': -25.5,
        'yaw': 0.1
    }
    
    # Vehicle GPS position
    vehicle_gps = {
        'latitude': 37.7749,    # San Francisco example
        'longitude': -122.4194,
        'altitude': 50.0        # 50 meters above ground
    }
    
    # Convert bounding box to GPS coordinates
    result = converter.bbox_center_to_gps(
        bbox=fire_detection_bbox,
        vehicle_attitude=vehicle_attitude,
        gimbal_attitude=gimbal_attitude,
        vehicle_gps=vehicle_gps,
        use_bbox_center=False  # Use bottom center for ground objects
    )
    
    if result:
        print("=== Fire Detection GPS Coordinates ===")
        print(f"Target GPS: {result['target_latitude']:.6f}, {result['target_longitude']:.6f}")
        print(f"Distance: {result['horizontal_distance_m']:.1f} meters")
        print(f"Bearing: {result['bearing_degrees']:.1f} degrees")
        print(f"Elevation angle: {result['elevation_angle_degrees']:.1f} degrees")
        print(f"Uncertainty: ±{result['uncertainty_m']:.1f} meters")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Bounding box: {result['bbox']}")
        print(f"Pixel coordinates: ({result['pixel_coordinates']['x']:.1f}, {result['pixel_coordinates']['y']:.1f})")
    else:
        print("Failed to convert bounding box to GPS coordinates")


if __name__ == "__main__":
    example_usage()