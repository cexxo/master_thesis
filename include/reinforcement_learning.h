#ifndef REINFORCEMENTEXO_H
#define REINFORCEMENTEXO_H

#include <vector>
#include "Point2D.h"
#include <string>
#include "ExoSagittalModel.h"

class ReinforcementExo : public ExoSagittalModel{

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
        std::vector<int> current_state = {1,1};
        std::vector<std::string> actions;
        std::vector<std::vector<std::vector<float>>> q_table;   //3D table, since the state is a x-y representation and the third dimension for each action
        std::vector<int> rewards;

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
        
        //set method of the qlearn part
        bool set_rewards(std::vector<int>);
        bool set_rewards(std::string, int);

        //get methods of the qLearn part
        float get_epsilon();
        float get_alpha();
        float get_gamma();
        int get_num_actions();
        int get_num_states();
        std::vector<int> get_current_state();
        std::vector<std::string> get_actions();
        std::vector<std::vector<std::vector<float>>> get_whole_table();
        float get_rewards(int, int, int);

        //set methods
        bool set_joint_angles(std::vector<float>);
        bool set_joint_limits(std::vector<Point2D>);
        bool set_positions(std::vector<Point2D>);

        //Reinforcement learning methods
        bool set_current_state(int x, int y);
        bool set_qLearner(float, float, float, int, int, int);
        float get_q(int, int, int);
        bool update_qTable(int, int, int, float);
        bool learnQ(int, int, int, float, float);
        int choose_action(int, int);
        bool learn(int, int, int, float, int, int);
        float executeAction(int, int, int, std::vector<int>&);

        //Learning Methods
        void startLearning(int, int, bool);
};
    
#endif 