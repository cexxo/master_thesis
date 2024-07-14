#include "Point2D.h"
#include <cmath>
#include <math.h>
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

float deg_to_rad(float degrees) {
    return degrees * (M_PI/180);
}


float norm(Point2D a, Point2D b) {
	return sqrt(pow(a.get()[0]-b.get()[0],2) + pow(a.get()[1]-b.get()[1],2));
}


float rad_to_deg(float rad){
	return rad*(180/M_PI);
}

float round( float arg, int digits ) {
    float retValue = arg * pow(10.0f,(float)digits);
    retValue = round(retValue);
    return retValue * std::pow(10.0f,(float)-digits);
}


void saveFullBodyTrajectories(std::vector<std::vector<Point2D>> bundle,  bool left_swing_leg, std::string filename) {

	std::ofstream outfile;
	outfile.open(filename + ".txt");
	int supp_knee_idx, supp_foot_idx, knee_idx, foot_idx, heel_idx, tip_idx; 
	if(left_swing_leg){
		supp_knee_idx=5;
		supp_foot_idx=6;
		knee_idx = 1;
		foot_idx = 2;
		heel_idx = 3;
		tip_idx = 4;
	}
	else {
		supp_knee_idx=1;
		supp_foot_idx=2;
		knee_idx = 5;
		foot_idx = 6;
		heel_idx = 7;
		tip_idx = 8;
	
	}
	for(int i=0; i< bundle[0].size();i++){
		outfile << 0 << " ";
			outfile << bundle[0][i].get()[0] << " " << bundle[0][i].get()[1] << " ";
			outfile << bundle[supp_knee_idx][i].get()[0] << " " << bundle[supp_knee_idx][i].get()[1] << " ";
			outfile << bundle[knee_idx][i].get()[0] << " " << bundle[knee_idx][i].get()[1] << " ";
			outfile << bundle[foot_idx][i].get()[0] << " " << bundle[foot_idx][i].get()[1] << " ";
			outfile << bundle[heel_idx][i].get()[0] << " " << bundle[heel_idx][i].get()[1] << " ";
			outfile << bundle[tip_idx][i].get()[0] << " " << bundle[tip_idx][i].get()[1] << " ";
			outfile << bundle[foot_idx][bundle[0].size()-1].get()[0] << " " << bundle[supp_foot_idx][i].get()[0] << " " << 0 << " " << 0 << std::endl;  
	
	}
	
	
}



std::vector<float> piecewise_cubic_traj_gen(float y_i, float z_i, float y_f, float z_f, float v_i, float v_f) { 
	std::vector<float> traj_coeffs(4);
	Eigen::Matrix4f A;
	Eigen::Vector4f b;
	A << pow(y_i,3),pow(y_i,2), y_i, 1, pow(y_f,3),pow(y_f,2), y_f, 1, 3*pow(y_i,2),2*y_i, 1, 0,   3*pow(y_f,2),2*y_f, 1, 0;
	b << z_i,z_f,v_i,v_f;
	Eigen::Vector4f x = A.colPivHouseholderQr().solve(b);
	for(int i=0;i<4;i++){
		traj_coeffs[i]= x[i];
	}
	return traj_coeffs;

}


std::vector<Point2D> compute_traj(std::vector<float> x, std::vector<float> traj_coeffs){ //generates points
	std::vector<Point2D> traj(x.size());
	float temp;
	float y;
	for(int i=0;i<x.size();i++){
		temp= pow(x[i],2);
		y= temp * x[i]* traj_coeffs[0] + temp * traj_coeffs[1] + x[i]* traj_coeffs[2] + traj_coeffs[3];
		traj[i] = Point2D(x[i], round(y, 5));
		
	}
	return traj;
}





