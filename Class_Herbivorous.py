import random
import numpy as np
import math
import cv2

height=int(1080/75)
width=int(1920/75)

def angle_lerp(current, target):
    # Compute the difference, normalized to (-π, π]
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    # Move a fraction of the way there
    return current + diff


class Herbivorous():
    grid_prey = np.empty((height, width), dtype=object)
    for i in range(grid_prey.shape[0]):
        for j in range(grid_prey.shape[1]):
            grid_prey[i, j] = []


    def __init__(self, height,width, Team, lifespan_repro, size_efficiency, wings=1, protect=1, mandi=1, illu=None, eye=1, speed=1, col=None, Pos=None, generation=None, intelect=None, type=None, grid_prey=grid_prey, age=0):
        #Things that will not change
        self.prop_rotation=0.5#Proportion of the last angle kept
        self.type=type
        self.limits=(width, height)
        self.col_or=col
        self.col=self.col_or
        self.ressources_intake=0.5#Proportion of food kept
        self.grid_prey=grid_prey
        self.eye=eye
        self.Illu=illu
        self.mandi=mandi
        self.protect=protect
        self.wings=wings
        self.P_flight=0.015*wings
        self.death_reason=None
        self.Team=Team

        self.proba_survie=np.arange(0,0.95,(0.95/21))[self.protect]
        self.lifespan_repro=lifespan_repro
        self.lifespan=7.5+20*(lifespan_repro)#How much secodn does the annimal live
        self.size_efficiency=size_efficiency
        self.repro=0.3+0.4*(1-lifespan_repro)# how frequently it will reproduce
        self.size=size_efficiency#Size of animal (small are eaten by big)

        min_efficiency=0.02
        max_efficiency=0.05
        self.efficiency = np.arange(min_efficiency,max_efficiency, (max_efficiency-min_efficiency)/21)[self.size_efficiency]# How fast do they consume the ressources they have (ressource/sec)


        # between 1 and 20
        self.dist_detection=self.eye/4# threshold dist
        self.intelect_or=intelect
        self.intelect=self.intelect_or*0.75# dirige vers gross bouffe
        self.speed_or=speed
        self.speed=np.arange(0,5,(5-1)/21)[self.speed_or]#how fast they move
        self.cur_speed = self.speed

        self.maturation=0.25#Prop age
        if self.mandi>0:
            self.ressources_comsumption=0.2+self.mandi/10
        else:
            self.ressources_comsumption=0

        self.flying=False

        if self.type=="C":
            self.eating_time=0

        if generation is None:
            self.generation=1
        else:
            self.generation=generation

        #Locomotion:
        if Pos is None:
            if self.type=="H":
                self.state=random.randint(0,1)#0=immobile, 1=moving
            else:
                self.state=1
            self.pos_X=random.random()*(width-2)+1
            self.pos_Y=random.random()*(height-2)+1
        else:
            self.state=1#0=immobile, 1=moving
            self.pos_X=Pos[0]
            self.pos_Y=Pos[1]

        self.angle=random.random()*math.pi*2

        self.alive = 1  # Track if it's alive
        self.age = age

        self.ressources_max = 1

        self.ressources_for_repro = 0.5
        self.ressources = (self.ressources_max / 2)

        self.dying=2

        #Only for testing!
        #self.repro = 0
        #self.lifespan = 300000


    def move_in_time(self, dt, grid):
        child = None
        if self.alive==0 and self.dying>0:
            if (self.death_reason=="Old age" or self.death_reason=="Not enought food"):

                self.col = [max(0, val-val*((2-self.dying)/2)) for val in self.col_or]

            else:
                x_mod = self.dying % 0.2
                if 0 <= x_mod <= 0.1 :
                    self.col=[max(0,val-150) for val in self.col_or]
                else:
                    self.col=self.col_or

            self.dying-=dt

        elif self.alive==0 and self.dying<=0:
            self.alive=-1

        else:
            if self.state:
                self.move_forward(dt)
                self.cur_speed = self.speed

            else:
                if self.type=="H":
                    self.eat_grass(grid,dt)
                if self.type=="C":
                    self.eating_time-=dt

            if ( self.type=="H") and random.random()<1*dt:
                self.state = 1 - self.state
            elif self.type=="C" and self.eating_time<=0:
                self.state=1

            escaping=False
            #We check if the animal fly:
            if self.wings>1 and self.flying<=0 and random.random()<self.P_flight*dt:
                self.flying=self.wings*0.1
                self.cur_speed=10

            if self.wings>1 and self.flying>0:
                self.cur_speed = 10
                self.flying-=dt

            else:
                #If it is a herbivore, we check for predatirs around:
                if self.type=="H" and random.random()<self.intelect:
                    sub_grid = self.grid_prey[max(0, round(self.pos_Y) - math.ceil(self.dist_detection)):min(
                        round(self.pos_Y) + math.ceil(self.dist_detection) + 1, int(self.limits[1])),
                               max(round(self.pos_X) - math.ceil(self.dist_detection), 0):min(
                                   round(self.pos_X) + math.ceil(self.dist_detection) + 1, int(self.limits[0]))]

                    possible_pred = []
                    for i in range(len(sub_grid)):
                        for j in range(len(sub_grid[i])):
                            possible_pred += [
                                (math.sqrt(math.pow(h.pos_X - self.pos_X, 2) + math.pow(h.pos_Y - self.pos_Y, 2)), h)
                                for h in sub_grid[i, j]
                                if h.alive==1 and h.size > self.size and
                                   math.sqrt(math.pow(h.pos_X - self.pos_X, 2) + math.pow(h.pos_Y - self.pos_Y,
                                                                                          2)) < self.dist_detection]

                    if len(possible_pred) > 0:
                        escaping = True
                        next_pred = min(possible_pred, key=lambda x: x[0])
                        dx = next_pred[1].pos_X - self.pos_X
                        dy = next_pred[1].pos_Y - self.pos_Y

                        target_angle = math.atan2(dy, dx) - math.pi

                        # print("Current: " + str(self.angle))
                        perfect_angle = angle_lerp(self.angle, target_angle)
                        # print("Perfect: " + str(perfect_angle))
                        # Add your random noise if you like:
                        self.angle = self.prop_rotation * (self.angle) + (1 - self.prop_rotation) * perfect_angle
                        self.cur_speed=self.speed
                        # print("Final: " + str(self.angle))
                        # print(" ")

                if self.state:#If we will move next: choosing angle.
                    if self.type=="H":
                        if random.random()<self.intelect:
                            if not escaping:
                                blurred_grid = grid.copy()

                                sub_grid = blurred_grid[max(0, round(self.pos_Y) - 1):min(
                                    round(self.pos_Y) + 1 + 1, int(self.limits[1])),
                                           max(round(self.pos_X) - 1, 0):min(
                                               round(self.pos_X) + 1 + 1, int(self.limits[0]))]

                                max_val = np.max(sub_grid)
                                indices = np.argwhere(sub_grid == max_val)
                                pos_x_corrected=self.pos_X-(max(round(self.pos_X) - 1, 0))
                                pos_y_corrected = self.pos_Y - (max(round(self.pos_Y) - 1, 0))

                                dists = np.sqrt((indices[:, 0] - pos_y_corrected) ** 2 + (indices[:, 1] - pos_x_corrected) ** 2)
                                # Pick the one with smallest distance
                                i = np.argmin(dists)
                                y, x = indices[i]

                                dx = x - (self.pos_X - max(round(self.pos_X) - 1, 0))
                                dy = y - (self.pos_Y - max(0, round(self.pos_Y) - 1))  # inverted Y because images grow downward

                                dist = math.hypot(dx, dy)
                                target_angle = math.atan2(dy, dx)

                                #print("Current: "+str(self.angle))

                                if dist > 0.5:
                                    perfect_angle = angle_lerp(self.angle, target_angle)
                                    #print("Perfect: " + str(perfect_angle))
                                    # Add your random noise if you like:
                                    self.angle = self.prop_rotation*(self.angle) + (1-self.prop_rotation)*perfect_angle
                                    #print("Final: " + str(self.angle))
                                    #print(" ")
                                else:
                                    self.cur_speed = 0
                        else:
                            #print("Random answer")
                            self.angle+=random.gauss(0, (0.1))


                if self.type=="C" and not self.eating_time>=dt and not self.ressources>0.75:
                    catched = False
                    #Does it catch a prey?
                    sub_grid = self.grid_prey[max(0, round(self.pos_Y) - 1):min(
                        round(self.pos_Y) + 1 + 1, int(self.limits[1])),
                               max(round(self.pos_X) - 1, 0):min(
                                   round(self.pos_X) + 1 + 1, int(self.limits[0]))]

                    possible_prey=[]
                    for i in range(len(sub_grid)):
                        for j in range(len(sub_grid[i])):
                            possible_prey+=[ (h.size, h) for h in sub_grid[i,j] if h.alive==1 and  h.Team!=self.Team and math.sqrt(math.pow(h.pos_X-self.pos_X,2)+math.pow(h.pos_Y-self.pos_Y,2))<(0.15)]

                    if len(possible_prey)>0:
                        next_prey = max(possible_prey, key=lambda x: x[0])

                        Pcatch_size=0.5-((self.size-next_prey[1].size)/20)/2
                        print(Pcatch_size)

                        if random.random() > next_prey[1].proba_survie and random.random()>Pcatch_size:
                            self.state=0
                            self.eating_time=5-((self.ressources_comsumption))
                            self.ressources=min(self.ressources+0.3+0.01*next_prey[0], self.ressources_max)
                            next_prey[1].die("Eaten by "+ self.Team+"'s")
                            catched=True
                        else:
                            self.eating_time=1.5
                            self.state=0

                    if not catched:
                        if random.random() < self.intelect:
                            sub_grid = self.grid_prey[max(0, round(self.pos_Y) - math.ceil(self.dist_detection)):min(
                                round(self.pos_Y) + math.ceil(self.dist_detection) + 1, int(self.limits[1])),
                                       max(round(self.pos_X) - math.ceil(self.dist_detection), 0):min(
                                           round(self.pos_X) + math.ceil(self.dist_detection) + 1, int(self.limits[0]))]

                            possible_prey = []
                            for i in range(len(sub_grid)):
                                for j in range(len(sub_grid[i])):
                                    possible_prey += [(math.sqrt(math.pow(h.pos_X - self.pos_X, 2) + math.pow(h.pos_Y - self.pos_Y, 2)), h)
                                                      for h in sub_grid[i, j]
                                                      if h.alive==1 and h.Team != self.Team and
                                                      math.sqrt(math.pow(h.pos_X - self.pos_X, 2) + math.pow(h.pos_Y - self.pos_Y, 2)) < self.dist_detection]

                            if len(possible_prey) > 0:
                                next_prey = min(possible_prey, key=lambda x: x[0])
                                dx = next_prey[1].pos_X - self.pos_X
                                dy = next_prey[1].pos_Y - self.pos_Y

                                target_angle = math.atan2(dy, dx)

                                #print("Current: " + str(self.angle))
                                perfect_angle = angle_lerp(self.angle, target_angle)
                                #print("Perfect: " + str(perfect_angle))
                                # Add your random noise if you like:
                                self.angle = self.prop_rotation*(self.angle) + (1-self.prop_rotation)*perfect_angle
                                #print("Final: " + str(self.angle))
                                #print(" ")
                            else:
                                self.angle+=random.gauss(0, (0.1))
                        else:
                            self.angle += random.gauss(0, (0.1))

                if self.age>(self.maturation*self.lifespan) and random.random()<self.repro*dt and self.ressources>self.ressources_for_repro:
                    child = self.reproduce()

            self.ressources -= (self.efficiency * dt)

            if self.age>self.lifespan:
                self.die("Old age")
                #print("Dying with "+str(self.age)+"sec old and "+str(self.ressources)+" food remaining")
            if self.ressources <= 0:
                self.die("Not enought food")
            self.age += dt#Animal aged one turn

        return(child)

    def reproduce(self):
        child = Herbivorous(self.limits[1],self.limits[0],lifespan_repro=self.lifespan_repro,
                            size_efficiency=self.size_efficiency, Pos=[self.pos_X,self.pos_Y], generation=self.generation+1,
                            col=self.col_or, intelect=self.intelect_or, type=self.type, speed=self.speed_or, eye=self.eye,
                            illu=self.Illu, mandi=self.mandi, protect=self.protect, wings=self.wings, Team=self.Team, age=0)

        self.ressources-=self.ressources_max/4

        return(child)

    def die(self, cause):
        try:
            self.grid_prey[round(self.pos_Y), round(self.pos_X)].remove(self)
        except:
            pass

        self.alive=0
        self.death_reason=cause


    def eat_grass(self, grid, dt):
        if grid[round(self.pos_Y), round(self.pos_X)]>self.ressources_comsumption * 255 * dt:
            self.ressources = min(self.ressources_intake * self.ressources_comsumption * dt + self.ressources,
                                  self.ressources_max)
            grid[round(self.pos_Y), round(self.pos_X)] = max(0, int(grid[round(self.pos_Y), round(self.pos_X)]) - (
                        self.ressources_comsumption * 255 * dt))

    def move_forward(self, dt):
        if self.state:
            try:
                self.grid_prey[round(self.pos_Y), round(self.pos_X)].remove(self)
            except:
                pass
            self.pos_X = self.pos_X + np.cos(self.angle) * self.cur_speed * dt
            self.pos_Y = self.pos_Y + np.sin(self.angle) * self.cur_speed * dt

            if self.pos_X >= self.limits[0] - 0.5:
                self.pos_X = self.limits[0] - 0.6
                self.angle += math.pi
            elif self.pos_X <= 0:
                self.pos_X = 0
                self.angle += math.pi

            if self.pos_Y >= self.limits[1] - 0.5:
                self.pos_Y = self.limits[1] - 0.6
                self.angle += math.pi
            elif self.pos_Y <= 0:
                self.pos_Y = 0
                self.angle += math.pi

            self.grid_prey[round(self.pos_Y), round(self.pos_X)].append(self)

            # print("X:" +str(self.pos_X))
            # print("Y:" + str(self.pos_Y))
            # print("lim X:" +str(self.limits[0]))
            # print("lim Y:" + str(self.limits[1]))
            # print("Age :"+str(self.age))
            # print("Ressources :"+str(self.ressources))
            # print("")


