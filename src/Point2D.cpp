#include "Point2D.h"
#include <iostream>

//Point Class Method Definition--------------------------------------------

Point2D::Point2D(){
	set(0.0, 0.0);
}


Point2D::~Point2D(){}



Point2D::Point2D(float x0, float y0){
	set(x0, y0);
}
void Point2D::set(float x0, float y0){
	x = x0;
	y = y0;
}
std::vector<float> Point2D::get(){;
	return std::vector<float> {x,y};
}


void Point2D::print(){
	std::cout << "(" << x << ", "
	<< y << ")"<< std::endl;
}

