export interface ConjunctionLatestResponse {
  event: Record<string, unknown>;
  orbit_points: {
    primary: number[][];
    secondary: number[][];
  };
  conjunction_point_eci_m: [number, number, number];
  b_plane: {
    miss_point_m: [number, number];
    covariance_ellipse_m: [number, number][];
    hbr_circle_m: [number, number][];
    pc: number;
    miss_distance_m: number;
  };
  pc_timeline: {
    time_to_tca_hours: number[];
    pc_reference: number[];
    pc_degraded: number[];
    threshold: number;
  };
}
