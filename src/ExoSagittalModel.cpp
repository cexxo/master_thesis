#include "ExoSagittalModel.h"
#include "GeneralUtilities.h"
#include "CubicTrajectoryGenerator2D.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <float.h>
#include <stdlib.h> 


//------------------UTILITY FUNCTIONS-------------------------------------------------------------





//------------EXO SAGITTAL MODEL CLASS FUNCTIONS-------------------------------------------------

ExoSagittalModel::ExoSagittalModel(float thigh_length, float shin_length, float heel_to_ankle_length, float tip_to_ankle_length) {
	this->thigh_length = thigh_length;
	this->shin_length = shin_length;
	this->heel_to_ankle_length = heel_to_ankle_length;
	this->tip_to_ankle_length = tip_to_ankle_length;
	setJointLimits(std::vector<Point2D>{Point2D(-45, 110), Point2D(-110, 0), Point2D(-45, 110), Point2D(-110, 0)});
	setAngles(std::vector<float>{0,0,0,0});
	
}

ExoSagittalModel::~ExoSagittalModel(){}



std::vector<float> ExoSagittalModel::getJointAngles() {
	return this->joint_angles;


}


std::vector<Point2D> ExoSagittalModel::getPositions() {
	return this->positions;
}


std::vector<Point2D> ExoSagittalModel::getJointLimits() {
	return this->joint_limits;
}


bool ExoSagittalModel::setJointLimits(std::vector<Point2D> limits) {
	if(limits.size()!=4) {
		std::cout<< "Length of the input angles' limits vector must be 4, in the following form: {right hip, right knee, left hip, left knee}. Unit of measurement is degrees. Each limit is defined by a lower and upper bound."<<std::endl;
		return false;
	}
	for(int i=0; i < 4 ; i++) {
		if(limits[i].get()[0]>limits[i].get()[1]){
			std::cout<< "Lower bound of limit with index " << i<< "is greater than upper bound. Make sure the lower bound is smaller than the upper one." <<std::endl;
			return false;
		
		}
		if(abs(limits[i].get()[0]) >= 360 || abs(limits[i].get()[1]) >= 360) {
			std::cout<< "Inconsistency detected in the value of the bounds. Make sure the bounds are between 0° and 360°(not included)" <<std::endl;
			return false;
		
		}
	}
	
	this->joint_limits = limits;
	return true;
}


bool ExoSagittalModel::setAngles(std::vector<float> joint_angles) {
	if(joint_angles.size()!=4) {
		std::cout<< "Length of the input angles vector must be 4, in the following form: {right hip, right knee, left hip, left knee}. Unit of measurement is degrees."<<std::endl;
		return false;
	
	}
	for(int i=0;i<4;i++){
		if( joint_angles[i]< this->joint_limits[i].get()[0] || this->joint_limits[i].get()[1]<joint_angles[i] ) {
			std::cout<< "Ciclo " << i << " è ok" <<  std::endl;
			std::cout<< "Input angle at index " << i << "is not valid. Bounds for joint "<< i << "are: ( " << this->joint_limits[i].get()[0] << " , " << this->joint_limits[i].get()[1] << " ) while the given value is " << joint_angles[i] << " ." <<std::endl;
			return false;
		}
	
	}
	this->joint_angles = joint_angles;
	updateForwardKinematics(); //must update positions, starting position of the chain is hip
	
	return true;
}
	
	
	
	
bool ExoSagittalModel::setPositions(std::vector<Point2D> positions){ // should only be called by exo
	if(positions.size()!=9) {
		std::cout<< "Length of the input position vector must be 9, in the following form: { hip, left knee, left foot, left heel, left tip, right knee, right foot, right heel, right tip }. Unit of measurement is meters."<<std::endl;
		return false;
	
	}
	this->positions = positions;
	updateInverseKinematics();
	return true;

}


std::vector<Point2D> ExoSagittalModel::calculateLegState(Point2D hip, Point2D ankle, bool left_leg) {
	std::vector<Point2D> res;
	float M = norm(hip, ankle);
	std::string s = left_leg==true?"left":"right";
	if(M-(this->thigh_length + this->shin_length)>0.03){
		std::cout<<"Kinematic constraint (leg length) not satisfied for "<< s << " leg." <<  std::endl;
		return res; //empty res means error
		
		
	}
	else{
		float temp = (pow(this->thigh_length,2) + pow(M,2) - pow(this->shin_length,2))/(2*this->thigh_length*M);
		if(temp>1) temp=1;
		if(temp<-1) temp=-1;
		float alfa = acos(temp);
		
		temp = (ankle.get()[0] - hip.get()[0])/M;
		if(temp>1) temp=1;
		if(temp<-1) temp=-1;
		float tilt = asin(temp);
		Point2D knee = Point2D( hip.get()[0] + this->thigh_length * sin(alfa + tilt) , hip.get()[1] - this->thigh_length * cos(alfa + tilt));
		
		//check inverse kinematics to see if there are inconsistencies with respect to joint limits
		float hip_ang = rad_to_deg(asin((knee.get()[0]- hip.get()[0])/this->thigh_length));
		//hip_ang = round(hip_ang,3);
		float knee_ang = rad_to_deg(asin((ankle.get()[0]- knee.get()[0])/this->shin_length) - deg_to_rad(hip_ang));
		knee_ang = round(knee_ang,0);
		hip_ang = round(hip_ang,0);
		if(left_leg){
			if(hip_ang < this->joint_limits[0].get()[0] || this->joint_limits[0].get()[1] < hip_ang ) {
				std::cout<<"Joint limit (left hip) not satisfied: Angle is " << hip_ang << std::endl;
				return res; //empty res means error
			}
			if(knee_ang < this->joint_limits[1].get()[0] || this->joint_limits[1].get()[1] < knee_ang ) {
				std::cout<<"Joint limit (left knee) not satisfied: Angle is " << knee_ang << std::endl;
				return res; //empty res means error
			}
		}
		else {
			if(hip_ang < this->joint_limits[2].get()[0] || this->joint_limits[2].get()[1] < hip_ang ) {
				std::cout<<"Joint limit (right hip) not satisfied: Angle is " << hip_ang << std::endl;
				return res; //empty res means error
			}
			if(knee_ang < this->joint_limits[3].get()[0] || this->joint_limits[3].get()[1] < knee_ang ) {
				std::cout<<"Joint limit (right knee) not satisfied: Angle is " << knee_ang << std::endl;
				return res; //empty res means error
			}
		
		}
		
		res.push_back(knee);
		temp = ( ankle.get()[0] - knee.get()[0])/this->shin_length;
		if(temp>1) temp=1;
		if(temp<-1) temp=-1;
		float epsilon = asin(temp);
		Point2D heel = Point2D(ankle.get()[0] - this->heel_to_ankle_length*cos(epsilon), ankle.get()[1] - this->heel_to_ankle_length * sin(epsilon));
		res.push_back(heel);
		Point2D tip = Point2D(ankle.get()[0] + this->tip_to_ankle_length *cos(epsilon), ankle.get()[1] + this->tip_to_ankle_length *sin(epsilon));
		res.push_back(tip);
	}
	return res;
	


}





bool ExoSagittalModel::updateInverseKinematics() {
	//left leg
	this->joint_angles[0] = asin((this->positions[1].get()[0]- this->positions[0].get()[0])/this->thigh_length);
	this->joint_angles[1] = asin((this->positions[2].get()[0]- this->positions[1].get()[0])/this->shin_length) - this->joint_angles[0];
	this->joint_angles[0] = rad_to_deg(this->joint_angles[0]);
	this->joint_angles[1] = rad_to_deg(this->joint_angles[1]);
	//right leg
	this->joint_angles[2] = asin((this->positions[5].get()[0]- this->positions[0].get()[0])/this->thigh_length);
	this->joint_angles[3] = asin((this->positions[6].get()[0]- this->positions[5].get()[0])/this->shin_length) - this->joint_angles[2];
	this->joint_angles[2] = rad_to_deg(this->joint_angles[2]);
	this->joint_angles[3] = rad_to_deg(this->joint_angles[3]);
	return true;

}


bool ExoSagittalModel::updateForwardKinematics() {

	if(this->positions.size()==0){
		this->positions.push_back(Point2D(0,0));
		this->positions.push_back(Point2D(this->positions[0].get()[0] + this->thigh_length * sin(deg_to_rad(this->joint_angles[0])), this->positions[0].get()[1] - this->thigh_length* cos(deg_to_rad(this->joint_angles[0])))); //left knee
		this->positions.push_back(Point2D(this->positions[1].get()[0] + this->shin_length* sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1])), this->positions[1].get()[1] - this->shin_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1])))); //left ankle
	
		this->positions.push_back(Point2D(this->positions[2].get()[0] - this->heel_to_ankle_length * sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1] + 90)), this->positions[2].get()[1] + this->heel_to_ankle_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1]+ 90)))); //left heel
	
		this->positions.push_back(Point2D(this->positions[2].get()[0] + this->tip_to_ankle_length * sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1] + 90)), this->positions[2].get()[1] - this->tip_to_ankle_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1]+ 90)))); //left tip
	
		this->positions.push_back(Point2D(this->positions[0].get()[0] + this->thigh_length * sin(deg_to_rad(this->joint_angles[2])), this->positions[0].get()[1] - this->thigh_length* cos(deg_to_rad(this->joint_angles[2])))); //right knee
		this->positions.push_back(Point2D(this->positions[5].get()[0] + this->shin_length* sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3])), this->positions[5].get()[1] - this->shin_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3])))); //right ankle
	
		this->positions.push_back(Point2D(this->positions[6].get()[0] - this->heel_to_ankle_length * sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3] + 90)), this->positions[6].get()[1] + this->heel_to_ankle_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3]+ 90)))); //right heel
	
		this->positions.push_back(Point2D(this->positions[6].get()[0] + this->tip_to_ankle_length * sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3] + 90)), this->positions[6].get()[1] - this->tip_to_ankle_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3]+ 90)))); //right tip
	}
	
	else{
		this->positions[1] = Point2D(this->positions[0].get()[0] + this->thigh_length * sin(deg_to_rad(this->joint_angles[0])), this->positions[0].get()[1] - this->thigh_length* cos(deg_to_rad(this->joint_angles[0]))); //left knee
		this->positions[2] = Point2D(this->positions[1].get()[0] + this->shin_length* sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1])), this->positions[1].get()[1] - this->shin_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1]))); //left ankle
	
		this->positions[3] = Point2D(this->positions[2].get()[0] - this->heel_to_ankle_length * sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1] + 90)), this->positions[2].get()[1] + this->heel_to_ankle_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1]+ 90))); //left heel
	
		this->positions[4] = Point2D(this->positions[2].get()[0] + this->tip_to_ankle_length * sin(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1] + 90)), this->positions[2].get()[1] - this->tip_to_ankle_length* cos(deg_to_rad(this->joint_angles[0]+ this->joint_angles[1]+ 90))); //left tip
	
		this->positions[5] = Point2D(this->positions[0].get()[0] + this->thigh_length * sin(deg_to_rad(this->joint_angles[2])), this->positions[0].get()[1] - this->thigh_length* cos(deg_to_rad(this->joint_angles[2]))); //right knee
		
		this->positions[6] = Point2D(this->positions[5].get()[0] + this->shin_length* sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3])), this->positions[5].get()[1] - this->shin_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3]))); //right ankle
	
		this->positions[7] = Point2D(this->positions[6].get()[0] - this->heel_to_ankle_length * sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3] + 90)), this->positions[6].get()[1] + this->heel_to_ankle_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3]+ 90))); //right heel
	
		this->positions[8] = Point2D(this->positions[6].get()[0] + this->tip_to_ankle_length * sin(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3] + 90)), this->positions[6].get()[1] - this->tip_to_ankle_length* cos(deg_to_rad(this->joint_angles[2]+ this->joint_angles[3]+ 90))); //right tip
	}
	return true;
}


bool ExoSagittalModel::updateHipPosition(Point2D hip) {
	this->positions[0] = hip;
	bool success = updateForwardKinematics();
	return success;
}


std::vector<std::vector<Point2D>> ExoSagittalModel::computeFullBodyTrajectory(std::vector<Point2D> hip_traj, std::vector<Point2D> left_ankle_traj, std::vector<Point2D> right_ankle_traj) {
	std::vector<std::vector<Point2D>> res;
	std::vector<Point2D> left_knee;
	std::vector<Point2D> left_heel;
	std::vector<Point2D> left_tip;
	std::vector<Point2D> right_knee;
	std::vector<Point2D> right_heel;
	std::vector<Point2D> right_tip;
	if(hip_traj.size() != left_ankle_traj.size() || hip_traj.size()!= right_ankle_traj.size()){
		std::cout<<"Hip trajectory and ankle trajectories must have the same size!"<< std::endl;
		return res; //empty res means error
	}
	std::vector<Point2D> temp_container;
	for(int i=0; i< hip_traj.size(); i++){
		temp_container = calculateLegState(hip_traj[i], left_ankle_traj[i], true);
		if(temp_container.size()==0) {
			std::cout<<"Hip trajectory and ankle trajectory are inconsistent for left leg (distance between them is greater than leg length or angle limits are not respected) at iteration "<< i << ". Aborting computation."<<std::endl;
			return res;
		
		}
		left_knee.push_back(temp_container[0]);
		left_heel.push_back(temp_container[1]);
		left_tip.push_back(temp_container[2]);
		temp_container = calculateLegState(hip_traj[i], right_ankle_traj[i], false);
		if(temp_container.size()==0) {
			std::cout<<"Hip trajectory and ankle trajectory are inconsistent for support leg (distance between them is greater than leg length or angle limits are not respected) at iteration "<< i <<". Aborting computation." <<std::endl;
			return res;
		}
		right_knee.push_back(temp_container[0]);
		right_heel.push_back(temp_container[1]);
		right_tip.push_back(temp_container[2]);
	}
	res.push_back(left_knee);
	res.push_back(left_heel);
	res.push_back(left_tip);
	res.push_back(right_knee);
	res.push_back(right_heel);
	res.push_back(right_tip);
	return res;
}


std::vector<std::vector<Point2D>> ExoSagittalModel::executeFullBodyTrajectory(std::vector<Point2D> hip_traj, std::vector<Point2D> left_ankle_traj, std::vector<Point2D> right_ankle_traj) {
	std::vector<std::vector<Point2D>> res;
	
	if(hip_traj.size() != left_ankle_traj.size() || hip_traj.size()!= right_ankle_traj.size()){
		std::cout<<"Hip trajectory and ankle trajectories must have the same size!"<< std::endl;
		return res; //empty res means error
	}
	if(hip_traj.size() ==0) {
		std::cout<<"Hip trajectory and foot trajectories are empty!"<< std::endl;
		return res;
	}
	if(norm(hip_traj[0], this->positions[0]) > 0.01) {
		std::cout<<"Discrepancy between current hip position and first point of the provided hip trajectory! Make sure these two points are the same when designing the hip trajectory."<< std::endl;
		return res;
	}

	if(round(norm(left_ankle_traj[0], this->positions[2]), 3) > 0) {
		std::cout<<"Discrepancy between current swing ankle position and first point of the provided swing ankle trajectory! Make sure these two points are the same when designing the ankle trajectory."<< std::endl;
		return res;
	}
	if(round(norm(right_ankle_traj[0], this->positions[6]), 3) > 0) {
		std::cout<<"Discrepancy between current support ankle position and provided support ankle position! Make sure these two points are the same when choosing the support ankle position."<< std::endl;
		return res;
	}
	
	res = computeFullBodyTrajectory(hip_traj, left_ankle_traj, right_ankle_traj);
	
	int final_index = hip_traj.size()-1;
	std::vector<Point2D> final_positions;
	std::vector<Point2D> temp;
	final_positions.push_back(hip_traj[final_index]);
	final_positions.push_back(res[0][final_index]);
	final_positions.push_back(left_ankle_traj[final_index]);
	final_positions.push_back(res[1][final_index]);
	final_positions.push_back(res[2][final_index]);
	final_positions.push_back(res[3][final_index]);
	final_positions.push_back(right_ankle_traj[final_index]);
	final_positions.push_back(res[4][final_index]);
	final_positions.push_back(res[5][final_index]);
	setPositions(final_positions);
	return res;
}


Point2D ExoSagittalModel::getHipPositionfromFootPlacement(std::vector<Point2D> feet_coordinates, float hip_lowering_constraint) {
	if(feet_coordinates.size()!=2) {
		std::cout<<"Input feet positions vector must have size 2!"<< std::endl;
		return Point2D(FLT_MIN,FLT_MIN);
	}
	if(round(feet_coordinates[0].get()[1], 3) != round(feet_coordinates[1].get()[1], 3)) {
		std::cout<<"Input feet positions don't have the same z-coordinate. This is a necessary condition to calculate hip height"<< std::endl;
		return Point2D(FLT_MIN,FLT_MIN);
	
	}
	float hip_y = (feet_coordinates[0].get()[0] + feet_coordinates[1].get()[0]) / 2;
	float hip_foot_dist_y = (feet_coordinates[0].get()[0] - feet_coordinates[1].get()[0])/2;
	float hip_z = sqrt(pow(this->thigh_length + this->shin_length ,2) - pow(hip_foot_dist_y,2));
	if((hip_z/(this->thigh_length + this->shin_length))< hip_lowering_constraint) {
		std::cout<<"Selected foot configuration makes hip drop below the limit percentage value ( " <<hip_lowering_constraint <<"% ). Change foot placement or hip_lowering_contrain percentage." << std::endl;
		return Point2D(FLT_MIN,FLT_MIN);
	
	}
	return Point2D(hip_y, hip_z + round(feet_coordinates[1].get()[1], 3));
	 

}
float ExoSagittalModel::computeHeightCCR(float knee_ang) {
	float a= atan(sin(deg_to_rad(knee_ang))/ (cos(deg_to_rad(knee_ang)) + this->shin_length/ this->thigh_length));
	float b= asin(this->shin_length/this->thigh_length * sin(a));
	float s= this->thigh_length + this->shin_length - (this->thigh_length*cos(b) + this->shin_length*cos(a));
	return s;	
}


	

std::vector<std::vector<Point2D>> ExoSagittalModel::computeStep(bool left_step, float step_length, Point2D step_peak, float step_duration, float hip_peak_time_coeff, float foot_peak_time_coeff, float support_knee_ang) {
	
	std::vector<std::vector<Point2D>> res;
	std::vector<Point2D> hip_traj;
	std::vector<Point2D> left_ankle_traj;
	std::vector<Point2D> right_ankle_traj;
	CubicTrajectoryGenerator2D gen = CubicTrajectoryGenerator2D();
	gen.setTimeVector(0, step_duration, "linear", 100);
	std::vector<Point2D> wps;
	std::vector<Point2D> vel;
	//Generate hip trajectory
	std::vector<float> tps{0,step_duration*hip_peak_time_coeff,step_duration};
	wps.push_back(Point2D(this->positions[0].get()[0], this->positions[0].get()[1])); //current hip_position
	std::cout<<"Hip Starting point is: (" << this->positions[0].get()[0] <<" , "<< this->positions[0].get()[1] << ")"<< std::endl;
	Point2D final_hip_pos;
	if(left_step)	{
		wps.push_back(Point2D(this->positions[6].get()[0], this->positions[6].get()[1]  + this->thigh_length + this->shin_length - computeHeightCCR(support_knee_ang))); //Hip midpoint is when its position is over the support foot
		std::cout<<"Hip Midpoint is: (" << this->positions[6].get()[0] <<" , "<< this->positions[6].get()[1] + this->thigh_length + this->shin_length -  computeHeightCCR(support_knee_ang) << ")"<< std::endl;
		Point2D final_swing_foot_pos = Point2D(this->positions[2].get()[0] + step_length, this->positions[2].get()[1]); 
		std::cout<<"Swing foot Endpoint is: (" << final_swing_foot_pos.get()[0] <<" , "<< final_swing_foot_pos.get()[1] << ")"<< std::endl;
		final_hip_pos = getHipPositionfromFootPlacement(std::vector<Point2D>{this->positions[6], final_swing_foot_pos}); //calculate final hip height from final feet configuration 
		std::cout<<"Hip Endpoint is: (" << final_hip_pos.get()[0] <<" , "<< final_hip_pos.get()[1] << ")"<< std::endl;
	}	
	else {
		wps.push_back(Point2D(this->positions[2].get()[0], this->positions[2].get()[1]  + this->thigh_length + this->shin_length - computeHeightCCR(support_knee_ang))); //Hip midpoint is when its position is over the support foot
		std::cout<<"Hip Midpoint is: (" << this->positions[2].get()[0] <<" , "<< this->positions[2].get()[1] + this->thigh_length + this->shin_length -  computeHeightCCR(support_knee_ang) << ")"<< std::endl;
		Point2D final_swing_foot_pos = Point2D(this->positions[6].get()[0] + step_length, this->positions[6].get()[1]);
		final_hip_pos = getHipPositionfromFootPlacement(std::vector<Point2D>{this->positions[2], final_swing_foot_pos});  //calculate final hip height from final feet configuration
		std::cout<<"Hip Endpoint is: (" << final_hip_pos.get()[0] <<" , "<< final_hip_pos.get()[1] << ")"<< std::endl;
	}
	if(final_hip_pos.get()[0] == FLT_MIN) {
		std::cout<<"Step length is too large to keep the hip at a reasonable height. Choose a smaller step length. Aborting." << std::endl;
		return res;
	}
	wps.push_back(final_hip_pos);
	vel.push_back(Point2D());
	vel.push_back(Point2D(0.05,0));
	vel.push_back(Point2D());
	gen.setWaypoints(wps, tps, vel);
	bool success = gen.generateTrajectory();
	if(!success){
		std::cout<<"Hip trajectory generation failed. Aborting." << std::endl;
		return res;
	}
	hip_traj = gen.getTrajectory();
	//Generate ankles trajectories
	vel.clear();
	//change velocities for swing ankle
	vel.push_back(Point2D(0, 1));
	vel.push_back(Point2D(0.1,0));
	vel.push_back(Point2D(0, -1));
	tps[1] = step_duration * foot_peak_time_coeff; //change timepoint for swing ankle
	wps.clear(); //reset waypiints for swing ankle
	if(left_step) {
		wps.push_back(Point2D(this->positions[2].get()[0], this->positions[2].get()[1]));
		wps.push_back(step_peak);
		wps.push_back(Point2D(this->positions[2].get()[0] + step_length, this->positions[2].get()[1]));
	}
	else {
		wps.push_back(Point2D(this->positions[6].get()[0], this->positions[6].get()[1]));
		wps.push_back(step_peak);
		wps.push_back(Point2D(this->positions[6].get()[0] + step_length, this->positions[6].get()[1]));
	}
	
	gen.setWaypoints(wps, tps, vel);
	success = gen.generateTrajectory();
	if(!success){
		std::cout<<"Swing ankle trajectory generation failed. Aborting." << std::endl;
		return res;
	}
	if(left_step) {
		left_ankle_traj = gen.getTrajectory();
		for(int i=0; i< left_ankle_traj.size(); i++) { right_ankle_traj.push_back(Point2D(this->positions[6].get()[0], this->positions[6].get()[1])); }
	}
	else {	
		right_ankle_traj = gen.getTrajectory();
		for(int i=0; i< right_ankle_traj.size(); i++) { left_ankle_traj.push_back(Point2D(this->positions[2].get()[0], this->positions[2].get()[1])); }
	}
	
	std::vector<std::vector<Point2D>> traj_bundle = computeFullBodyTrajectory(hip_traj, left_ankle_traj, right_ankle_traj);
	
	if(traj_bundle.size()==0) {
		std::cout<<"Error during body trajectory computation. Aborting." << std::endl;
		return res;
	}
	
	res.push_back(hip_traj);
	res.push_back(traj_bundle[0]);
	res.push_back(left_ankle_traj);
	res.push_back(traj_bundle[1]);
	res.push_back(traj_bundle[2]);
	res.push_back(traj_bundle[3]);
	res.push_back(right_ankle_traj);
	res.push_back(traj_bundle[4]);
	res.push_back(traj_bundle[5]);	
	return res;
	
}



std::vector<std::vector<Point2D>> ExoSagittalModel::executeStep(bool left_step, float step_length, Point2D step_peak , float step_duration, float hip_peak_time_coeff, float foot_peak_time_coeff, float support_knee_ang) {
	std::vector<std::vector<Point2D>> res = computeStep(left_step, step_length, step_peak, step_duration, hip_peak_time_coeff, foot_peak_time_coeff, support_knee_ang);
	int final_index = res[0].size()-1;
	std::vector<Point2D> final_positions;
	for(int i=0;i< res.size(); i++) final_positions.push_back(res[i][final_index]);
	setPositions(final_positions);
	return res;
}
	



std::vector<std::vector<Point2D>> ExoSagittalModel::executeLeftStep(float step_length, float step_height, float step_duration) {
	
	float initial_y_swing_ankle = this->positions[2].get()[0];
	
	Point2D midpoint = Point2D((2*initial_y_swing_ankle + step_length)/2 , this->positions[2].get()[1] + step_height);

	std::vector<std::vector<Point2D>> res = executeStep(true, step_length, midpoint, step_duration);
	
	return res;


}
	
	
	
std::vector<std::vector<Point2D>> ExoSagittalModel::executeRightStep(float step_length, float step_height, float step_duration) {

	float initial_y_swing_ankle = this->positions[6].get()[0];
	
	Point2D midpoint = Point2D((2*initial_y_swing_ankle + step_length)/2 , this->positions[6].get()[1] + step_height);

	std::vector<std::vector<Point2D>> res = executeStep(false, step_length, midpoint, step_duration);
	
	return res;

}
	
	

std::vector<std::vector<Point2D>> ExoSagittalModel::executeFeetAlignment(float step_height, float step_duration) {
	std::vector<std::vector<Point2D>> res;
	bool left_step;
	Point2D midpoint;
	if(round(this->positions[2].get()[0],3) == round(this->positions[6].get()[0],3)) {
		std::cout<<"Feet already aligned. Aborting." << std::endl;
		return res;
	}
	else if(round(this->positions[2].get()[0],3) < round(this->positions[6].get()[0],3)) {
		left_step = true;
		midpoint = Point2D(this->positions[6].get()[0]-0.05, this->positions[6].get()[1] + step_height);
	}
	else {
		left_step = false;
		midpoint = Point2D(this->positions[2].get()[0]-0.05, this->positions[2].get()[1] + step_height);
	}
	std::cout << "Step length for alignment: " << abs(this->positions[6].get()[0] - this->positions[2].get()[0]) << std::endl;
	res = executeStep(left_step, abs(this->positions[6].get()[0] - this->positions[2].get()[0]) , midpoint, step_duration, 0.5, 0.7);
	
	if(res.size() == 0) {
		std::cout<<"Failure during step execution. Aborting." << std::endl;
		return res;
	}
	return res;


}
	






