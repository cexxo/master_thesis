#include "reinforcement_learning.h"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include "Point2D.h"
#include "ExoSagittalModel.h"

ReinforcementExo::ReinforcementExo(float thight_length, float shin_length) : ExoSagittalModel(0.5, 0.5 , 0.09, 0.11){
    this->thight_length = thight_length;
    this->shin_length = shin_length;
}

ReinforcementExo::ReinforcementExo(
    float thight_length, float shin_length, std::vector<Point2D> positions
    ) : ExoSagittalModel(0.5, 0.5 , 0.09, 0.11){
    this->thight_length = thight_length;
    this->shin_length = shin_length;
    this->positions = positions;
}

ReinforcementExo::~ReinforcementExo(){}

//GETTERS

float ReinforcementExo::get_shin(){
    return this->shin_length;
}

float ReinforcementExo::get_thight(){
    return this->thight_length;
}

std::vector<float> ReinforcementExo::get_joint_angles(){
    return this->joint_angles;
}

std::vector<Point2D> ReinforcementExo::get_joint_limits(){
    return this->joint_limits;
}

std::vector<Point2D> ReinforcementExo::get_positions(){
    return this->positions;
}

float ReinforcementExo::get_alpha(){
    return this->alpha;
}

float ReinforcementExo::get_gamma(){
    return this->gamma;
}

float ReinforcementExo::get_epsilon(){
    return this->epsilon;
}

int ReinforcementExo::get_num_actions(){
    return this->num_actions;
}

int ReinforcementExo::get_num_states(){
    return this->num_states;
}

std::vector<std::string> ReinforcementExo::get_actions(){
    return this->actions;
}

std::vector<int> ReinforcementExo::get_current_state(){
    return this->current_state;
}

float ReinforcementExo::get_rewards(int x, int y, int action){
    return this->q_table[x][y][action];
}


float ReinforcementExo::get_q(int x, int y, int action){
    if(x >= this->q_table.size() || x < 0){
        std::cout << "trying to access x: " << x << std::endl;
        std::cout << "get_q: x_axis error: accessing invalid state" << std::endl  << std::endl;
        return -1;
    }else if(y >= this->q_table[0].size() || y < 0){
        std::cout << "trying to access y: " << y << std::endl;
        std::cout << "get_q: y_axis error: accessing invalid state" << std::endl  << std::endl;
        return -1;
    }else if(action >= this->q_table[0][0].size() || action < 0){
        std::cout << "trying to access k: " << action << std::endl;
        std::cout << "get_q: k_axis error: accessing invalid state" << std::endl  << std::endl;
        return -1;
    }
    return this->q_table[x][y][action];
}

std::vector<std::vector<std::vector<float>>> ReinforcementExo::get_whole_table(){
    return this->q_table;
}

//SETTERS

bool ReinforcementExo::set_joint_angles(std::vector<float>new_angles){
    if(new_angles.size() != this->get_joint_angles().size()){
        std::cout << "The size of the input vector does not match the size of the angle vector" << std::endl;
        return false;
    }
    for(int i = 0; i<this->get_joint_angles().size();i++) this->joint_angles[i] = new_angles[i];
    return true;
}

bool ReinforcementExo::set_joint_limits(std::vector<Point2D>new_limits){
    if(new_limits.size() != this->get_joint_limits().size()){
        std::cout << "The size of the input vector does not match the size of the joint limit vector" << std::endl;
        return false;
    }
    for(int i = 0; i < this->get_joint_limits().size();i++) this->joint_limits[i] = new_limits[i];
    return true;
}

bool ReinforcementExo::set_positions(std::vector<Point2D>new_positions){
    if(new_positions.size() != this->get_positions().size()){
        std::cout << "The size of the input vector does not match the size of the positions vector" << std::endl;
        return false;
    }
    for(int i = 0; i < this->get_positions().size();i++) this->positions[i] = new_positions[i];
    return true;
}


bool ReinforcementExo::set_qLearner(float alpha, float gamma, float epsilon, int num_actions, int width, int height){
    this->alpha = alpha;
    this->gamma = gamma;
    this->epsilon = epsilon;
    this->num_actions = num_actions;
    std::vector<std::vector<std::vector<float> > > v(width, std::vector<std::vector<float> >(height, std::vector<float>(num_actions)));
    this->q_table = v;
    for(int k = 0; k < this->get_whole_table()[0][0].size();k++){
        for(int i = 0; i < this->get_whole_table().size();i++){
            for (int j = 0; j < this->get_whole_table()[0].size();j++){
                //std::cout << (float) rand()/RAND_MAX << std::endl;
                //this->q_table[i][j][k] = ((float) rand()/RAND_MAX)*-1;
                this->q_table[i][j][k] = (1 + rand() % 2)*-1;
            }
        }
    }
    /*this->q_table[9][9][0] = 2;
    this->q_table[9][10][0] = 2;
    this->q_table[19][19][0] = 10;*/
    return true;
}

bool ReinforcementExo::set_rewards(std::vector<int> new_rewards){
    if(new_rewards.size() != this->get_actions().size()){
        std::cout << "the size of the reward value does not match the desired one: " << new_rewards.size() << " != " << this->get_actions().size() << std::endl;
        return false;
    }
    this->rewards = new_rewards;
    return true;
}

bool ReinforcementExo::set_current_state(int x, int y){
    if(x >= this->q_table.size() || x < 0){
        std::cout << "set_current_state: x_axis error: accessing invalid state" << std::endl;
        return false;
    }else if(y >= this->q_table[0].size() || y < 0){
        std::cout << "set_current_state: y_axis error: accessing invalid state" << std::endl;
        return false;
    }else{
        this->current_state[0]= x;
        this->current_state[1]= y;
        return true;
    }
}

//Reinforcement Learning methods

bool ReinforcementExo::update_qTable(int x, int y, int action, float new_value){
    if(x >= this->q_table.size() || x < 0){
        std::cout << "trying to access x: " << x << std::endl;
        std::cout << "update table: x_axis error: accessing invalid state" << std::endl << std::endl;
        return false;
    }else if(y >= this->q_table[0].size() || y < 0){
        std::cout << "trying to access y: " << y << std::endl;
        std::cout << "update table: y_axis error: accessing invalid state" << std::endl << std::endl;
        return false;
    }else if(action >= this->q_table[0][0].size() || action < 0){
        std::cout << "trying to access k: " << action << std::endl;
        std::cout << "update table: k_axis error: accessing invalid state" << std::endl << std::endl;
        return false;
    }
    else{
        this->q_table[x][y][0] = new_value;
        return true;
    }
}

bool ReinforcementExo::learnQ(int x, int y, int action, float reward, float value){
    float old_value = this->get_q(x, y, action);
    if (old_value == -1){
        this->update_qTable(x, y, action,reward);
        return false;
    }else{
        this->update_qTable(x, y, action,(old_value + this->alpha * (value - old_value)));
    }
    return true;
}

int ReinforcementExo::choose_action(int x, int y){
    std::vector<float> q;
    int index = -1;
    std::vector<float> max_q;
    std::vector<int> dummy;
    for (int i = 0; i < this->num_actions;i++){
        //std::cout << "here" << std::endl;
        if (this->executeAction(x,y,i,dummy) != -1){
            //std::cout << "here2" << std::endl;
            q.push_back(this->executeAction(x,y,i,dummy));
        }
    }
    float max = -100000;
    for (int i = 0; i < q.size();i++){
        if(q[i] > max){
            max = q[i];
        }
    }
    for(int i = 0; i < q.size();i++){
        if(q[i] == max)
            max_q.push_back(i);
    }
    float random = rand()/RAND_MAX;
    if (random < this->epsilon){
        float min = 10000;
        for(int i = 0; i < q.size();i++){
            if (q[i] < min){
                min = q[i];
            }       
        }
        max = abs(max);
        min = abs(min);
        float magnitude;
        if(max >= min) magnitude = max;
        else magnitude = min;
        for (int i = 0; i < this->actions.size();i++){
            q[i] +=  rand()/RAND_MAX*magnitude - 0.5*magnitude;
        }
        max = -10000;
        if(max_q.size() == 1){
            for (int i = 0; i < q.size();i++){
                if(q[i] > max){
                    max = q[i];
                    index = i;
                }
            }
            std::cout << "action choice " << index << std::endl;
        }else{
            index = (rand()%max_q.size());
            std::cout << "action choice " << index << std::endl;;
        }

    }
    return index;
}

bool ReinforcementExo::learn( int x1, int y1, int action, float reward, int x2, int y2){
    float new_max = -10000;
    for(int i = 0; i < this->num_actions;i++){
        if(new_max < this->get_q(x2, y2, 0)){
            new_max = this->get_q(x2, y2, 0);
        }
    }
    this->learnQ(x1, y1, action, reward, reward + this->gamma * new_max);
    return true;
}


//0 = up    1 = up-right    2 = right   3 = right-bottom    4 = bottom  5 = bottom-left     6 = left    7= top-left 
float ReinforcementExo::executeAction(int x, int y, int action, std::vector<int>& new_state){      //THIS FUNCTION IS GONNA BE THE FULCRUM OF THE CODE. I HAVE TO THINK IT CORRECTLY
    bool forward = true;
    bool backward = true;
    bool top = true;
    bool down = true;
    if(x == 19){
        forward = false;
    }
    if(x == 0){
        backward = false;
    }
    if(y == 19){
        down = false;
    }
    if(y == 0){
        top = false;
    }
    switch (action)
    {
    case 0:
        if(top){
            new_state = {x,y-1};
            return this->q_table[x][y-1][0];
        }
        break;
    case 1:
        if(top && forward){
            new_state = {x+1,y-1};
            return this->q_table[x+1][y-1][0];
        }
        break;
    case 2:
        if(forward){
            new_state = {x+1,y};
            return this->q_table[x+1][y][0];
        }
        break;
    case 3:
        if(forward && down){
            new_state = {x+1,y+1};
            return this->q_table[x+1][y+1][0];
        }
        break;
    case 4:
        if(down){
            new_state = {x,y+1};
            return this->q_table[x][y+1][0];
        }
        break;
    case 5:
        if(down && backward){
            new_state = {x-1,y+1};
            return this->q_table[x-1][y+1][0];
        }
        break;
    case 6:
        if(backward){
            new_state = {x-1,y};
            return this->q_table[x-1][y][0];
        }
        break;
    case 7:
        if(backward && top){
            new_state = {x-1,y+1};
            return this->q_table[x-1][y+1][0];
        }
        break;
    default:
        std::cout << "EXECUTE ACTION: something went wrong, impossible to choose action " << std::endl;
        return -1;
        break;
    }
    return -1;
}

void ReinforcementExo::startLearning(int numEpisodes, int numSteps, bool display){
    float epsilon_discount = 0.999;
    float highestReward = 0;
    for (int i = 0; i < numEpisodes;i++){
        std::cout << "Starting episode: " << i << std::endl;
        float cumulatedReward = 0;
        bool done = false;
        if(this->get_epsilon() > 0.05)
            this->epsilon *= epsilon_discount;
        std::vector<int> state = this->get_current_state();
        std::cout << state[0] << " " << state[1] << std::endl;
        std::vector<int> new_state = {-1,-1};
        std::vector<int> executed_actions;
        for (int j = 0; j < numSteps; j++){
            std::cout << "Start step: " << j << std::endl;
            std::cout << state[0] << " " << state[1] << std::endl;
            int action = this->choose_action(state[0], state[1]);
            executed_actions.push_back(action);
            int reward = 0;
            this->executeAction(state[0],state[1],action,new_state);
             if(new_state[0] > state[0]){
                if(new_state[1] >= state[1]){
                    reward = 1;
                }else{
                    reward = -1;
                }
            }else if(new_state[0] == 19 && new_state[1] == 19){
                reward = 10;
            }else{
                reward = -1;
            }
            cumulatedReward += reward;
            /*std::cout << state[0] << " " << state[1] << std::endl;
            std::cout << new_state[0] << " " << new_state[1] << std::endl << std::endl << std::endl << std::endl;
            std::cout << action << std::endl << std::endl << std::endl << std::endl;*/
            if (highestReward < cumulatedReward)
                highestReward = cumulatedReward;
            this->learn(state[0],state[1], action, reward, new_state[0], new_state[1]);
            state = new_state;
            if(new_state[0] == 19 && new_state[1] == 19){
                std::cout << "finished at iteration: " << j << std::endl;
                break;
            }
            //Execute the action
            //This action should give me a reward according to where i found myself afterwards
            //I increment the current cumulated reward
            //I calculate the next state accoridng to how the robot found itself
            //I update the state and proceed to the next iteration
        }
        this->action_sequences.push_back(executed_actions);
        if(display){
            for(int k = 0; k < this->get_whole_table()[0][0].size();k++){
                    std::cout << "table " << k << std::endl;
                    for(int i = 0; i < this->get_whole_table().size();i++){
                        for (int j = 0; j < this->get_whole_table()[0].size();j++){
                            std::cout << this->get_whole_table()[i][j][k] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    std::cout << std::endl;
                    std::cout << std::endl;
            }
        }
    }
}

std::vector<std::vector<int>> ReinforcementExo::get_executed_actions(){
    return this->action_sequences;
}