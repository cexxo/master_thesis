#ifndef TRAJECTORYGENERATOR2D_H
#define TRAJECTORYGENERATOR2D_H


#include "Point2D.h"
#include <vector>


//Trajectory Generator Abstract Class-----------------------------------------------------------------------------

class TrajectoryGenerator2D{
	public:
	virtual std::vector<Point2D> generateTrajectory()=0;
};

#endif
