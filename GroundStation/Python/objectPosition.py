import numpy as np
from scipy.spatial.transform import Rotation as R
import pymap3d


class ObjectPosition:
    def __init__(self, USE_ELEVATION = False, MAP = None):
        self.USE_ELEVATION = USE_ELEVATION
        self.MAP = MAP
        if USE_ELEVATION:
            if MAP is None:
                raise ValueError("MAP must be provided when USE_ELEVATION is True")
            self.getObjectPosition = self.getObjectPosition_elevPixel
        else:
            self.getObjectPosition = self.getObjectPosition_flatPixel
        self.altHome = 0

    def getObjectPosition_flatPixel(self, x_image, y_image, f, cx, cy, gb_pitch, gb_yaw, lat_drone, lon_drone, alt_drone):
        # Position of the object in the image expressed in the camera frame
        r_c = np.array([[f],[x_image - cx],[y_image - cy]])

        # Rotation matrix from camera frame to world frame
        C_cw = R.from_euler('ZY', [gb_yaw, gb_pitch], degrees=True).as_matrix()

        # Computing the intersection with ground
        alpha = alt_drone/np.matmul(C_cw, r_c)[2][0]
        r_w = alpha*np.matmul(C_cw, r_c) + np.array([[0], [0], [-alt_drone]])
        
        # Compute position in earth frame
        # animalPosition = pymap3d.ned2geodetic(r_w[0][0], r_w[1][0], 0, lat_drone, lon_drone, 0)[0:2]
        firePosition = pymap3d.ned2geodetic(r_w[0][0], r_w[1][0], 0, lat_drone, lon_drone, alt_drone)[0:2]

        # results = {"animalLat":animalPosition[0], "animalLon":animalPosition[1]}
        results = {"FireLat": firePosition[0], "FireLon": firePosition[1]}
        return results

    def getObjectPosition_elevPixel(self, x_image, y_image, f, cx, cy, gb_pitch, gb_yaw, lat_drone, lon_drone, alt_drone_Home):
        # print(f"Call to getObjectPosition_elevPixel with {x_image}, {y_image}, {f}, {cx}, {cy}, {gb_pitch}, {gb_yaw}, {lat_drone}, {lon_drone}, {alt_drone_Home}")
        # Position of the object in the image expressed in the camera frame
        r_c = np.array([[f],[x_image - cx],[y_image - cy]])

        # Rotation matrix from camera frame to world frame
        C_cw = R.from_euler('ZY', [gb_yaw, gb_pitch], degrees=True).as_matrix()

        # Find intersection with sruface
        alt_drone_DEM = alt_drone_Home + self.altHome
        animalLat, animalLon, animalAlt = self.MAP.findIntersectionWithSurface(np.array([lat_drone, lon_drone, alt_drone_DEM]), np.matmul(C_cw,r_c))

        results = {"animalLat":animalLat, "animalLon":animalLon, "animalAlt":animalAlt}
        return results
    
    def setAltHome(self, homeLat, homeLon):
        self.altHome = self.MAP.getElevation(homeLat, homeLon, method="rayTrace")
        print(f"Home altitude set to {self.altHome}")

    
if __name__ == "__main__":
    # localiser = ObjectPosition(USE_ELEVATION=False)
    
    # # Example usage
    # pos = localiser.getObjectPosition_flatPixel(320, 240, 773.8937725706451, 411.5987780481727, 209.30710180683946, -2.6, 177.9, 53.6634006283764, -2.6564995639206708, 3.5)
    # print(pos)
    print("This is not a standalone script")
    