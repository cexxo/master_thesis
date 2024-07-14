#include "CubicTrajectoryGenerator2D.h"
#include "Point2D.h"
#include "GeneralUtilities.h"
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <ros/ros.h>




//------------CUBIC TRAJECTORY GENERATOR 2D CLASS FUNCTIONS---------------------------------------------------


CubicTrajectoryGenerator2D::CubicTrajectoryGenerator2D() {} //constructor doesn't do anything

CubicTrajectoryGenerator2D::~CubicTrajectoryGenerator2D() {} //constructor doesn't do anything



bool CubicTrajectoryGenerator2D::setTimeVector(float t_i, float t_f, std::string spacing_type, int n_points){
	if(spacing_type=="linear"){
		if(n_points<2) return false; //failure, can't be possible to have less than 2 points
		else{
			float delta= (t_f - t_i) / (n_points-1);
			for(int i=0; i< n_points;i++){
				this->timeVector.push_back(t_i + delta*i);
			}
		}
	}
	return true;
}





bool  CubicTrajectoryGenerator2D::setWaypoints(std::vector<Point2D> waypoints, std::vector<float> timepoints, std::vector<Point2D> velocities){
	this->wayPoints.clear();
	this->velocities.clear();
	this->timePoints.clear();
	for(int i=0; i< timepoints.size()-1; i++) {
		if(timepoints[i]> timepoints[i+1]){
			std::cout<<"Timepoints vector must be an increasing sequence! The condition is not met. "<<std::endl;
			return false;
		}
	}
	if(this->timeVector.size()==0){ //DA MODIFICARE! CREO IL TIME VECTOR A PARTIRE DAI TIMEPOINTS
		std::cout<<"Time vector not set! Define a time vector for the trajectory before setting the waypoints."<<std::endl;
		return false;
	}
	if(waypoints.size()!=timepoints.size()) {
		std::cout<<"Mismatch between number of waypoints and number of timepoints!"<<std::endl;
		return false;
	}
	if(waypoints.size()!=velocities.size()) {
		std::cout<<"Mismatch between number of waypoints and number of velocities!"<<std::endl;
		return false;
	}
	if(timepoints[0]< this->timeVector[0] || timepoints[timepoints.size()-1] > this->timeVector[this->timeVector.size()-1]) { //if timepoints exceed the range of the time vector 
		std::cout<<"Timepoints exceed the range of the time vector! Modify the timepoints or the time vector to make sure the condition is met."<<std::endl;
		return false;
	}
	for(int i=0;i < waypoints.size(); i++) {
		this->wayPoints.push_back(waypoints[i]);
		this->velocities.push_back(velocities[i]);
	
	}
	//find most similar timepoint in the time vector
	for(int i=0; i< timepoints.size(); i++){
		float diff=1000;
		float old_diff = 1000;
		int idx = 0;
		do{
			if(diff!=old_diff) old_diff=diff;
			diff= abs(timepoints[i] - this->timeVector[idx]);
			idx++;
		}
		while(diff<old_diff && diff!=0 && idx< this->timeVector.size());
		this->timePoints.push_back(this->timeVector[idx-1]);
	
	}
	return true;
}


bool CubicTrajectoryGenerator2D::modifyWaypoint(int index, Point2D waypoint, float time, Point2D velocity){
	if(this->wayPoints.size()==0){
		std::cout<<"Waypoints not set! You can't modify a waypoint if they are not set."<<std::endl;
		return false;
	}
	if(this->wayPoints.size()<index) {
		std::cout<<"Index exceeds the size of the waypoints! Make sure to select a valid index."<<std::endl;
		return false;
	}
	bool inconsistent=false;
	if(index==0) {
		if(this->timePoints[index+1]<time){
			inconsistent=true;
		
		}
	}
	else if(index == this->wayPoints.size()-1){
		if(this->timePoints[index-1]>time){
			inconsistent=true;
		
		}
	}
	else {
		if(this->timePoints[index-1]>time || this->timePoints[index+1]<time){
			inconsistent=true;
		
		}
	}
	if(inconsistent){
		std::cout<<"Time inconsistency detected! Make sure that the new timepoint is between the previous and the next one."<<std::endl;
		return false;
	}
	else {
		this->wayPoints[index] = waypoint;
		this->velocities[index] = velocity;
		float diff=1000;
		float old_diff = 1000;
		int idx =0;
		do{
			if(diff!=old_diff) old_diff=diff;
			diff= abs(time - this->timeVector[idx]);
			idx++;
		}
		while(diff<old_diff);
		this->timePoints[index] = this->timeVector[idx-1];
		return true;
	}
}



bool CubicTrajectoryGenerator2D::generateTrajectory(){
	
	
	std::vector<float> cubic_coeffs;
	std::vector<Point2D> x_t; //x(t)
	std::vector<float> temp;
	std::vector<Point2D> piecewise_y_x; // piecewise y(x)
	std::vector<Point2D> y_x; // y(x)
	std::vector<float> t;
	int j=0;
	for(int i=0; i<this->wayPoints.size()-1;i++){
		//isolate portion of time vector between timepoints
		if(t.size()!=0) {
			float old_t = t[t.size()-1];
			t.clear();
			t.push_back(old_t);
		}
		if(j==0){
			while(this->timeVector[j] != this->timePoints[0]) j++; //align with first time point
		}
		//add all points between first and last timepoint (first and last included)
		while(this->timeVector[j] != this->timePoints[i+1]) {
			t.push_back(this->timeVector[j]);
			j++;
		}
		if(i==wayPoints.size()-2) t.push_back(this->timeVector[j]); //add last timepoint at the last iteration of i
		//generate x(t)
		std::vector<float> p_i =  this->wayPoints[i].get();
		std::vector<float> p_f =  this->wayPoints[i+1].get();
		std::vector<float> v_i =  this->velocities[i].get();
		std::vector<float> v_f =  this->velocities[i+1].get();
		float f = p_i[0];
		cubic_coeffs= piecewise_cubic_traj_gen(t[0], p_i[0], t[t.size()-1], p_f[0], v_i[0], v_f[0]);
		x_t = compute_traj(t , cubic_coeffs);
		cubic_coeffs= piecewise_cubic_traj_gen(x_t[0].get()[1], p_i[1], x_t[x_t.size()-1].get()[1], p_f[1], v_i[1], v_f[1]);
		temp.clear();
		for(int k=0; k<x_t.size();k++) temp.push_back(x_t[k].get()[1]);
		
		piecewise_y_x = compute_traj(temp , cubic_coeffs);
		//ADD PIECEWISE TRAJECTORY TO THE TOTAL TRAJECTORY
		if(i<wayPoints.size()-2)	y_x.insert(y_x.end(),piecewise_y_x.begin(),piecewise_y_x.end()-1);
		else				y_x.insert(y_x.end(),piecewise_y_x.begin(),piecewise_y_x.end());	
	}

	this->trajectory = y_x;
	return true;
}


std::vector<Point2D> CubicTrajectoryGenerator2D::getTrajectory(){
	return this->trajectory;
}


std::vector<float> CubicTrajectoryGenerator2D::getTimeVector(){
	return this->timeVector;
}

