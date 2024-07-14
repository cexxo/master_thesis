#include "reinforcement_learning.h"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include "Point2D.h"

ReinforcementExo::ReinforcementExo(float thight_length, float shin_length){
    this->thight_length = thight_length;
    this->shin_length = shin_length;
}

ReinforcementExo::ReinforcementExo(
    float thight_length, float shin_length, std::vector<Point2D> positions
    ){
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

int ReinforcementExo::get_current_state(){
    return this->current_state;
}

std::vector<int> ReinforcementExo::get_rewards(){
    return this->rewards;
}

int ReinforcementExo::get_index_from_action(std::string action){
    bool found = false;
    int index = 0;
    for (int i = 0; i < this->actions.size();i++){
        if(action == this->actions[i]){
            found = true;
            index = i;
        }
    }
    if(!found){
        std::cout << "trying to access unavailable action" << std::endl;
        return -1;
    }
    return index;
}

float ReinforcementExo::get_q(int state, std::string action){
    if(state >= this->q_table.size()){
        std::cout << "trying to access unavailable state" << std::endl;
        return -1;
    }
    int index = this->get_index_from_action(action);
    if (index != -1)
        return this->q_table[state][index];
    else
        return -1;
}

std::vector<std::vector<float>> ReinforcementExo::get_whole_table(){
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

bool ReinforcementExo::set_qLearner(
    float alpha, float gamma, float epsilon, int num_actions, std::vector<std::string> actions,int num_states
    ){
    if (num_actions != actions.size()){
        std::cout << "the number of actions must be equal to the actions vector size" << std::endl;
        return false;
    }
    this->alpha = alpha;
    this->gamma = gamma;
    this->epsilon = epsilon;
    this->num_actions = num_actions;
    this->num_states = num_states;
    this->current_state = 0;
    for(int i = 0; i < this->num_actions;i++)
        this->actions.push_back(actions[i]);
    std::vector<std::vector<float>> temp(num_states);
    this->q_table = temp;
    for(int i = 0; i < q_table.size();i++){
        for(int j = 0; j < num_actions; j++){
            q_table[i].push_back(0);
        }
    }
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

bool ReinforcementExo::set_rewards(std::string action, int new_reward){
    if(this->get_index_from_action(action) != -1){
        this->rewards[this->get_index_from_action(action)] = new_reward;
        return true;
    }else{
        std::cout << "no action " << action << " found, impossible set new value" << std::endl;
        return false;
    }
}

//Reinforcement Learning methods

bool ReinforcementExo::update_qTable(int state, std::string action, float new_value){
    if(state >= this->q_table.size()){
        std::cout << "trying to access unavailable state" << std::endl;
        return false;
    }
    int index = this->get_index_from_action(action);
    if(index != -1){
        this->q_table[state][index] = new_value;
        return true;
    }
    return false;
}

bool ReinforcementExo::learnQ(int state, std::string action, float reward, float value){
    float old_value = this->get_q(state,action);
    if (old_value == -1){
        this->update_qTable(state,action,reward);
        return false;
    }else{
        this->update_qTable(state,action,(old_value + this->alpha * (value - old_value)));
    }
    return true;
}

std::string ReinforcementExo::choose_action(int state){
    std::vector<float> q;
    int index = -1;
    for (int i = 0; i < this->actions.size();i++){
        q.push_back(this->get_q(state,this->actions[i]));
    }
    float max = -100000;
    for (int i = 0; i < q.size();i++){
        if(q[i] > max){
            max = q[i];
        }
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
        for (int i = 0; i < q.size();i++){
            if(q[i] > max){
                max = q[i];
                index = i;
            }
        }

    }
    return this->actions[index];
}

bool ReinforcementExo::learn(std::string action, int state1, float reward, int state2){
    float new_max = -10000;
    for(int i = 0; i < this->actions.size();i++){
        if(new_max < this->get_q(state2,this->actions[i])){
            new_max = this->get_q(state2,this->actions[i]);
        }
    }
    this->learnQ(state1, action, reward, reward + this->gamma * new_max);
    return true;
}

void ReinforcementExo::startLearning(int numEpisodes, int numSteps){
    float epsilon_discount = 0.999;
    for (int i = 0; i < numEpisodes;i++){
        std::cout << "Starting episode: " << i << std::endl;
        float cumulatedReward = 0;
        bool done = false;
        if(this->get_epsilon() > 0.05)
            this->epsilon *= epsilon_discount;
        int state = this->get_current_state();
        for (int j = 0; j < numSteps; j++){
            std::cout << "Start step: " << j << std::endl;
            std::string action = this->choose_action(state);
        }
    }
}