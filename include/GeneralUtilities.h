#ifndef GENERALUTILITIES_H
#define GENERALUTILITIES_H

#include "Point2D.h"
#include <vector>
#include <string>

float deg_to_rad(float degrees);
float norm(Point2D a, Point2D b);
float rad_to_deg(float rad);
float round(float arg, int digits);
void saveFullBodyTrajectories(std::vector<std::vector<Point2D>> bundle, bool left_swing_leg, std::string filename);


std::vector<float> piecewise_cubic_traj_gen(float y_i, float z_i, float y_f, float z_f, float v_i, float v_f);
std::vector<Point2D> compute_traj(std::vector<float> x, std::vector<float> traj_coeffs);
std::vector<Point2D> piecewise_cubic_poly(std::vector<Point2D> waypoints, std::vector<float> indep_var, std::vector<float> wp_velocities);



#endif
