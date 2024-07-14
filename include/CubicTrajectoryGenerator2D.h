#ifndef CUBICTRAJECTORYGENERATOR2D_H
#define CUBICTRAJECTORYGENERATOR2D_H

#include "TrajectoryGenerator2D.h"
#include "Point2D.h"
#include <string>
#include <vector>



//Pieceise Cubic Trajectory Generator Class-----------------------------------------------------------------------------


class CubicTrajectoryGenerator2D {
	private:
	std::vector<float> timePoints;
	std::vector<float> timeVector;
	std::vector<Point2D> velocities;
	std::vector<Point2D> wayPoints;
	std::vector<Point2D> trajectory;
	public:
	CubicTrajectoryGenerator2D();
	~CubicTrajectoryGenerator2D();
	bool setTimeVector(float t_i, float t_f, std::string spacing_type, int n_points=0); //spacing_type defines the rule which creates the intermediate points
	bool setWaypoints(std::vector<Point2D> waypoints, std::vector<float> timepoints, std::vector<Point2D> velocities);
	bool modifyWaypoint(int index, Point2D waypoint, float time, Point2D velocity);
	bool generateTrajectory();
	std::vector<Point2D> getTrajectory();
	std::vector<float> getTimeVector();
};


#endif
