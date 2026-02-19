import random as rd
import torch
from DQN import deep_q_network
import train

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class GridWorld:
    goal_reward = 10
    step_reward = -1
    obstacles_reward = -50

    def __init__(self, size:tuple[int], start:tuple[int], goal:tuple[int], obstacles_type, obstacles:list[tuple[int]]=[], nb_obstacles=0):
        '''
        :param size: (x,y) size of the grid world
        :type size: tuple[int]
        :param start: (x,y) starting position
        :type start: tuple[int]
        :param goal: (x,y) goal position
        :type goal: tuple[int]
        :param obstacles_type: "random" or "fixed"
        :param obstacles: if obstacles_type is "fixed", a list of (x,y) positions of obstacles
        '''

        self.gridsize = size
        self.start = start
        self.position = start
        self.goal = goal

        if obstacles_type == "random":
            self.obstacles = self.generate_random_obstacles(nb_obstacles)
        elif obstacles_type == "fixed":
            self.obstacles = obstacles
        
    def show(self):
        print("="*self.gridsize[0]*2)
        grid = [["." for i in range(self.gridsize[0])] for i in range(self.gridsize[1])]
        # grid[self.position[1]][self.position[0]] = "o"
        grid[self.start[1]][self.start[0]] = "S"
        grid[self.goal[1]][self.goal[0]] = "G"
        for obstacle in self.obstacles:
            grid[obstacle[1]][obstacle[0]] = "X"
        for row in grid:
            print(" ".join(row))
    
    def generate_random_obstacles(self, nb_obstacles=0):
        possible_pos = [(x,y) for x in range(self.gridsize[0]) for y in range(self.gridsize[1]) if (x,y) != self.start and (x,y) != self.goal]
        obstacles = []
        for i in range(nb_obstacles):
            choice = rd.choice(possible_pos)
            obstacles.append(choice)
            possible_pos.remove(choice)
        return obstacles
    
    def is_valid_move(self, action):
        match action:
            case x if  x=="up" or x==0:
                new_pos = (self.position[0], self.position[1]-1)
            case x if x=="down" or x==1:
                new_pos = (self.position[0], self.position[1]+1)
            case x if x=="right" or x==2:
                new_pos = (self.position[0]+1, self.position[1])
            case x if x=="left" or x==3:
                new_pos = (self.position[0]-1, self.position[1])
        if new_pos[0]<0 or new_pos[0]>=self.gridsize[0] or new_pos[1]<0 or new_pos[1]>=self.gridsize[1]:
            return False
        if new_pos in self.obstacles:
            return False
        else:
            return True
    
    def move(self, action): 
        '''
        :param action: "up", "down", "right", "left"
        '''

        if True or self.is_valid_move(action):
            # reward = self.step_reward
            match action:
                case x if  x=="up" or x==0:
                    self.position = (self.position[0], self.position[1]-1)
                case x if x=="down" or x==1:
                    self.position = (self.position[0], self.position[1]+1)
                case x if x=="right" or x==2:
                    self.position = (self.position[0]+1, self.position[1])
                case x if x=="left" or x==3:
                    self.position = (self.position[0]-1, self.position[1])
            # if self.is_goal():
            #     reward += self.goal_reward
            #     return self.position, self.goal_reward+self.step_reward
            # else:
            return self.position, self.get_reward()


    
    def is_goal(self):
        return self.position == self.goal
    
    def reset(self):
        self.position = self.start
    
    def get_reward(self):
        if self.is_goal():
            return self.goal_reward
        elif self.position in self.obstacles:
            return self.obstacles_reward
        else:
            return self.step_reward
    
    def get_state(self):
        return self.position
    
    def get_random_state(self):
        return (rd.randint(0, self.gridsize[0]-1), rd.randint(0, self.gridsize[1]-1))
    

def show_policy(policy_net, env):


    print("="*env.gridsize[0]*2)
    grid = [["." for i in range(env.gridsize[0])] for i in range(env.gridsize[1])]
    grid[env.start[1]][env.start[0]] = "S"
    
    for y in range(env.gridsize[1]):
        for x in range(env.gridsize[0]):
            state = torch.tensor((x,y), dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state).argmax().item()
            match action:
                case 0:
                    grid[y][x] = "↑"
                case 1:
                    grid[y][x] = "↓"
                case 2:
                    grid[y][x] = "→"
                case 3:
                    grid[y][x] = "←"
    
    grid[env.goal[1]][env.goal[0]] = "G"
    for obstacle in env.obstacles:
        grid[obstacle[1]][obstacle[0]] = "X"
    for row in grid:
        print(" ".join(row))
    

if __name__ == "__main__":
    try:
        # env = GridWorld((5,5), (0,0), (4,4), "random", nb_obstacles=3)
        env = GridWorld((5,5), (0,0), (4,4), "fixed", [(1,1), (2,2), (3,3)])
        # env = GridWorld((100, 50), (0, 0), (99, 49), "random", nb_obstacles=500)

        # # Q_network = deep_q_network(2, 4)

        policy_net = train.train(env)


        # # env.show()
        # # env.move("right")
        # # env.show()
        # policy_net = deep_q_network(2, 4).to(device)
        # policy_net.load_state_dict(torch.load("policy_net.pth"))
    finally:
        # torch.save(policy_net.state_dict(), "policy_net.pth")
        
        print(f"Q of {env.get_state()} : {policy_net(torch.tensor(env.get_state(), dtype=torch.float32, device=device)).unsqueeze(0).tolist()}")
        env.show()
        show_policy(policy_net, env)