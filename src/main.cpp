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
    /*std::vector<std::string> actions = {"up", "up-right", "right", "right_bottom", "bottom" , "bottom_left", "left", "top_left"};
    std::cout << exo.set_qLearner(0.1,0.7,0.9,actions.size(),actions,5) << std::endl;  //In the simplest case i have 5 states: before, on top and after the obstacle.
    std::cout << exo.get_whole_table().size() << std::endl;
    std::cout << exo.get_whole_table()[0].size() << std::endl;
    std::cout << exo.get_q(2,"prova") << std::endl;
    exo.update_qTable(2,"prova",12);
    std::cout << exo.get_q(2,"prova") << std::endl;*/
    std::cout << exo.set_qLearner(0.1, 0.7, 0.9, 3, 20, 20) << std::endl;
    /*for(int k = 0; k < exo.get_whole_table()[0][0].size();k++){
        std::cout << "table " << k << std::endl;
        for(int i = 0; i < exo.get_whole_table().size();i++){
            for (int j = 0; j < exo.get_whole_table()[0].size();j++){
                std::cout << exo.get_whole_table()[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }*/
    exo.startLearning(10,10);
    //MISSING TO SEE IF LEARNQ WORKS
    return 0;
}