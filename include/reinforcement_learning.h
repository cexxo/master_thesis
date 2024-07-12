#ifndef REINFORCEMENTEXO_H
#define REINFORCEMENTEXO_H

#include <vector>
#include "Point2D.h"
#include <string>

class ReinforcementExo{

    private:
        //EXO CONSTANTS
        float shin_length;
        float thight_length;
        std::vector<float> joint_angles = {0,0,0,0};
        std::vector<Point2D> joint_limits = {Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0)};
        std::vector<Point2D> positions = {Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0), Point2D(0,0)};

        //QLEARN CONSTANTS(LATER TO BE MOVED TO A SPECIFIC CLASS, MAYBE)
        float epsilon;
        float alpha;
        float gamma;
        int num_actions;
        int num_states;
        std::vector<std::string> actions;
        std::vector<std::vector<float>> q_table;

    public:
        ReinforcementExo(float thight_length, float shin_length);
        ReinforcementExo(float, float, std::vector<Point2D>);
        ~ReinforcementExo();

        //get methods
        float get_shin();
        float get_thight();
        std::vector<float> get_joint_angles();
        std::vector<Point2D> get_joint_limits();
        std::vector<Point2D> get_positions();


        //set methods
        bool set_joint_angles(std::vector<float>);
        bool set_joint_limits(std::vector<Point2D>);
        bool set_positions(std::vector<Point2D>);

        //Reinforcement learning methods
        bool set_qLearner(float, float, float, int, std::vector<std::string>, int);
        int get_index_from_action(std::string);
        float get_q(int, std::string);
        std::vector<std::vector<float>> get_whole_table();
        bool update_qTable(int, std::string, float);
        bool learnQ(int, std::string, float, float);
};
    
#endif 