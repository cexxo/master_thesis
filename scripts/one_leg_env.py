import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

class leftLeg():
    def __init__(self, shin_lenght, thight_lenght, hx, hy, kx, ky, tx, ty):
        self.shin_length = shin_lenght
        self.thight_lenght = thight_lenght
        self.hx = hx
        self.hy = hy
        self.kx = kx
        self.ky = ky
        self.tx = tx
        self.ty = ty

    def __sub__(self, other):
        return (self.tx-other.x, self.ty-other.y)
    
    def action(self,choice):
        if choice >= 0 and choice <=30:
            self.move_thight(choice)
        elif choice >= -30 and choice <0:
            self.move_thight(choice)
        elif choice >= 30.1 and choice <=60:
           self.move_shin(choice - 30)
        elif choice >= -60 and choice <-30:
            self.move_shin(choice + 30)
    
    def move_thight(self, rho = False):
        r = self.thight_lenght
        r = 10
        if not rho :
            rho = 30*np.random.random() - 30 
            rho *= 3.14/180 
            self.ky += int(r * np.sin(rho))
            self.kx += int(r * np.cos(rho))
            print("not rho")
            #self.tx += int(r * np.sin(rho))
            #self.ty += int(r * np.cos(rho))
        else:
            rho *= 3.14/180
            self.ky = self.ty - int(r * np.cos(rho)) 
            self.kx = self.tx - int(r * np.sin(rho))
            print(f"rho  {rho}        {r * np.cos(rho)}")
            #self.tx += int(r * np.sin(rho))
            #self.ty += int(r * np.cos(rho))
        print(f"{self.ky}       {self.kx}")

    def move_shin(self, rho = False):
        r = self.shin_length
        if not rho :
            rho = 30*np.random.random() - 30
            rho *= 3.14/180
            self.ty -= int(r * np.sin(rho))
            self.tx -= int(r * np.cos(rho))
        else:
            rho *= 3.14/180
            self.ty = int(r * np.sin(rho))
            self.tx = int(r * np.cos(rho))
        print(f"{self.ky}       {self.kx}")


class Blob:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

JOINT_1 = 1
JOINT_2 = 2
JOINT_3 = 3
OBSTACLE_N = 4
GOAL_N = 5
MIDPOINT_N = 6
ENV_SIZE = 100
d ={ 1: (255,175,0),
     2: (255,100,0),
     3: (255,25,0),
     4 : (0, 0, 255),
     5 : (0, 255, 0),
     6 : (20, 255, 20) 
}
player = leftLeg(5, 5, 8, int(ENV_SIZE/2) - 6, 14, int(ENV_SIZE/2) - 6, 20, int(ENV_SIZE/2) - 6)    #Stop with magic numbers
show = True
goal = Blob(ENV_SIZE - 1, int(ENV_SIZE/2) + 3)
obstacle = Blob(ENV_SIZE - 1, int(ENV_SIZE/2))
for i in range(-180,-120,10):
    if show:
        env = np.zeros((ENV_SIZE, ENV_SIZE, 3), dtype=np.uint8)     
        #env[goal.x][goal.y] = d[GOAL_N]                                                    
        env[obstacle.x][obstacle.y] = d[OBSTACLE_N]
        env[goal.x][goal.y] = d[GOAL_N]                 
        #env[mid_point.x][mid_point.y] = d[MIDPOINT_N]
        h_point = (player.hy, player.hx)
        k_point = (player.kx, player.ky)
        t_point = (player.tx, player.ty)
        cv2.line(env, k_point, t_point, (255, 255, 255), 1)  
        cv2.circle(env, (player.tx,player.ty), 1, (255,0,255))
        cv2.circle(env, (player.kx,player.ky), 1, (255,0,255))
        #cv2.line(env, k_point, t_point, (255, 255, 255), 1) 
        #env[player.hx][player.hy] = d[JOINT_1]                        
        #env[player.kx][player.ky] = d[JOINT_2]                        
        #env[player.tx][player.ty] = d[JOINT_3] 
        img = Image.fromarray(env, 'RGB')                           
        img = img.resize((800, 800))                      
        cv2.imshow("image", np.array(img))   
        #player.action(i/10)
        player.move_thight(i)
        cv2.waitKey(500)
cv2.waitKey(0)
cv2.destroyAllWindows() 
input()                  