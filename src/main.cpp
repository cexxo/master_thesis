#include "reinforcement_learning.h"
#include <iostream>
#include <string>

int main(){
    ReinforcementExo exo = ReinforcementExo(1.0,1.0);
    std::cout << exo.get_thight() << " " << exo.get_thight() << std::endl;
    std::cout << exo.set_joint_angles({1,2,3,4,5,6}) << std::endl; 
    //std::cout << exo.set_joint_limits({Point2D(1,2), Point2D(2,2), Point2D(0,0), Point2D(4,0)}) << std::endl;
    //std::cout << exo.set_positions({Point2D(0,0), Point2D(0,1), Point2D(0,2), Point2D(5,0),Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(9,0),Point2D(0,0)}) << std::endl;
    std::vector<Point2D> temp = exo.get_positions();
    /*for(int i=0;i<temp.size();i++)
        std::cout << temp[i].get()[0] << " " << temp[i].get()[1] << std::endl;*/
    std::vector<std::string> actions = {"prova","prova2"};
    std::cout << exo.set_qLearner(1,1,1,2,actions,3) << std::endl;
    std::cout << exo.get_whole_table().size() << std::endl;
    std::cout << exo.get_whole_table()[0].size() << std::endl;
    std::cout << exo.get_q(2,"prova") << std::endl;
    exo.update_qTable(2,"prova",12);
    std::cout << exo.get_q(2,"prova") << std::endl;
    //MISSING TO SEE IF LEARNQ WORKS
    return 0;
}