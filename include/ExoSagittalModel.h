#ifndef EXOSAGITTALMODEL_H
#define EXOSAGITTALMODEL_H
#include <vector>
#include "Point2D.h"


//Exo Sagittal Model class--------------------------------------------------------------------------------

class ExoSagittalModel {

	private: //METHODS
	bool updateInverseKinematics(); //based on the updated joint positions, updates joint angles
	bool updateForwardKinematics(); //based on the updated joint angles, updates joint positions
	bool setPositions(std::vector<Point2D> positions); //set positions
	std::vector<Point2D> calculateLegState(Point2D hip, Point2D ankle, bool left_leg);
	float computeHeightCCR(float knee_ang);
	std::vector<std::vector<Point2D>> computeFullBodyTrajectory(std::vector<Point2D> hip_traj, std::vector<Point2D> left_ankle_traj, std::vector<Point2D> right_ankle_traj); //compute trajectory starting from current configuration
	std::vector<std::vector<Point2D>> executeFullBodyTrajectory(std::vector<Point2D> hip_traj, std::vector<Point2D> left_ankle_traj, std::vector<Point2D> right_ankle_traj);
	Point2D getHipPositionfromFootPlacement(std::vector<Point2D> feet_y_coordinates, float com_lowering_constraint=0.9); //assumes hip y-coordinate in the middle between the two feet y-coordinates and straight legs (for maximum stability). We also don't want hip height to fall under 0.9* leg length (this leaves for a 10% change of height w.r.t. maximum extension, only reached when hip height = leg length)
	
	public: //METHODS
	ExoSagittalModel(float thigh_length, float shin_length, float heel_to_ankle_length, float tip_to_ankle_length);
	~ExoSagittalModel();
	bool updateHipPosition(Point2D hip);
	bool setAngles(std::vector<float> joint_angles);
	bool setJointLimits(std::vector<Point2D> limits);
	std::vector<float> getJointAngles();
	std::vector<Point2D> getJointLimits();
	std::vector<Point2D> getPositions();
	std::vector<std::vector<Point2D>> computeStep(bool left_step, float step_length, Point2D step_peak, float step_duration, float hip_peak_time_coeff=0.5, float foot_peak_time_coeff=0.3, float support_knee_ang=5);
	std::vector<std::vector<Point2D>> executeStep(bool left_step, float step_length, Point2D step_peak, float step_duration, float hip_peak_time_coeff=0.5, float foot_peak_time_coeff=0.3, float support_knee_ang=5);
	std::vector<std::vector<Point2D>> executeLeftStep(float step_length, float step_height, float step_duration);
	std::vector<std::vector<Point2D>> executeRightStep(float step_length, float step_height, float step_duration);
	std::vector<std::vector<Point2D>> executeFeetAlignment(float step_height, float step_duration); //brings the rear foot in line with the front one
	
	
	
	
	
	private: //VARIABLES
	float thigh_length, shin_length, heel_to_ankle_length, tip_to_ankle_length; // Exo dimensions in the sagittal plane (width of the link is not currently considered) 
	std::vector<float> joint_angles; // {Left Hip, Left Knee, Right Hip, Right Knee} , in degrees
	std::vector<Point2D> joint_limits; //angular range for each joint
	std::vector<Point2D> positions; // { Hip, Left knee, Left foot, left heel, left tip, right knee, right foot, right heel, right tip } , in meters on the sagittal plane (y,z)
};
#endif
