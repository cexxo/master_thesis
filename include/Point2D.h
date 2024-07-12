#ifndef POINT2D_H
#define POINT2D_H
#include <vector>

//2D Point class--------------------------------------------------------------------------------

class Point2D{
	private:
	float x, y;
	public:
	Point2D();
	~Point2D();
	Point2D (float x0, float y0);
	void set(float x0, float y0);
	std::vector<float> get();
	void print();
};

#endif



